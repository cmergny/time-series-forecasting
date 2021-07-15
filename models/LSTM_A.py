# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:37:09 2021

@author: Cyril
"""

### IMPORTS

import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from models.LSTM_AE import Encoder, Decoder

### CLASSES

    
class Attention(nn.Module):
    """ Following Bahdanau et al. 2015
    Neural machine translation by jointly learning to align and translate
    """
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, input_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.linear3 = nn.Linear(input_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, out_e, hidden_d):
        """S, N, H the Source length, Batch size and Hidden size"""
        # out_e.shape = (S, N, H)
        hidden_d = hidden_d.repeat(out_e.shape[0], 1, 1) 
        # hidden_d.shape = (1, N, H) -> (S, N, H)
        a = self.linear3(self.tanh(self.linear1(hidden_d) + self.linear2(out_e))) # (S, N, 1) 
        alpha = self.softmax(a) # (S, N, 1))
        c = torch.einsum("sne,snh->enh", alpha, out_e) # c.shape = (1, N, H)
        return c, alpha
      
class LSTM_Attention(nn.Module):
    """
    Use the two Encoder and Decoder classes to train a LSTM neural network
    Can also make predictions once the NN is trained
    """
    def __init__(self, input_size, hidden_size):
        """ Initialising variables with param and Encoder/Decoder classes"""
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.attention = Attention(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size)
        

    def forward(self, x, target_len):
        # Initialise outputs
        outputs = torch.zeros(target_len,  x.shape[1], x.shape[2]).to(x.device) # (T, N, E)
        self.alphas = torch.zeros(x.shape[0], x.shape[1], target_len).to(x.device) # (S, N, T)

        # Call Encoder 
        out_e, hidden_e, cell_e = self.encoder(x)
        # Initialise Decoder
        input_d = x[-1, :, :].unsqueeze(0) # shape(N, E)
        hidden_d, cell_d = hidden_e, cell_e
        
        # Iterate by values to predict
        for t in range(target_len):
            # Call Decoder
            hidden_d, alpha = self.attention(out_e, hidden_d)
            self.alphas[:, :, t] = alpha.squeeze(-1)
            out_d, hidden_d, cell_d = self.decoder(input_d, hidden_d, cell_d)
            outputs[t] = out_d
            input_d = out_d.unsqueeze(0)
        return(outputs)
    
    def save(self, path='last_model'):
        "Saves model to the saved_models folder"
        torch.save(self.state_dict(), path)
        #print(f'Saved model to {path}')
        return None
