# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:37:09 2021

@author: Cyril
"""

### IMPORTS


import torch
import torch.nn as nn
from models.LSTM_AE import Encoder, Decoder
from models.LSTM_A import Attention

### MODEL


class MultiScaleAttention(nn.Module):
    """ Multi-scale Context Based Attention for Dynamic Music Emotion Prediction
    Ma et al. 2017
    """
    def __init__(self, hidden_size) -> None:
        super().__init__()
        #self.linear2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(2*hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, C, hidden_d):
        # C.shape = (E, N, H)
        hidden_d = hidden_d.repeat(C.shape[0], 1, 1) # (E, N, H)
        energy = self.tanh(self.linear(torch.cat((C, hidden_d), dim=2))) # (E, N, 1)
        #e = torch.abs(energy[:, 0, 0])
        #print(torch.max(e), torch.min(e), torch.sum(e))
        #print(self.softmax(e))
        #a = input()
        beta = self.softmax(energy) # (E, N, 1)
        c = torch.einsum("enl,enh->lnh", beta, C) # c.shape = (1, N, H)
        return(c, beta)      
    
class MultiScaleLSTMA(nn.Module):
    
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoders =  nn.ModuleList([Encoder(1, hidden_size) for _ in range(input_size)])
        #self.decoders = nn.ModuleList([Decoder(1, hidden_size) for _ in range(input_size)])
        self.decoder = Decoder(1, hidden_size)
        self.multiscale_attention = MultiScaleAttention(hidden_size)
        
    def forward(self, x, target_len):
        """Pass all modes into differents lstm encoders.
           Then for each mode:
                - Compute the attention bw the encoder output of the mode
                and the encoder outputs of all modes
                - Decode the attention vector to predict the next p steps
        """       
        # x.shape = (S, N, E)
        
        # Initialise outputs and self.hiddens self.cells
        outputs = torch.zeros(target_len,  x.shape[1], x.shape[2]).to(x.device) # (S, N, E)
        self.hiddens = torch.zeros(self.input_size, x.shape[1], self.hidden_size).to(x.device)  # (E, N, H)
        self.cells = torch.zeros(self.input_size, x.shape[1], self.hidden_size).to(x.device)  # (E, N, H)
        self.alphas = torch.zeros(self.input_size, x.shape[1], self.input_size).to(x.device) # (E, N, E)
        
        # Call encoder for all modes seperately
        for i, encoder in enumerate(self.encoders):
            xi = x[:, :, i].unsqueeze(-1) # (S, N, 1)
            _, self.hiddens[i], self.cells[i] = encoder(xi) #(1, N, H)
        
        # For each mode
        for i in range(self.input_size):
            # Init Decoder inputs
            input_d = x[-1, :, i].unsqueeze(-1).unsqueeze(0)
            hidden_d, cell_d = self.hiddens[i].unsqueeze(0), self.cells[i].unsqueeze(0)
            # Call Attention bw hiddens of all modes and current mode
            hidden_d, alpha = self.multiscale_attention(self.hiddens, hidden_d)
            # Get attention weights
            self.alphas[:, :, i] = alpha.squeeze(-1) # alpha.shape = (E, N, 1) -> (E, N)
            # Call Decoder p times
            for t in range(target_len):
                #out_d, hidden_d, cell_d = self.decoders[i](input_d, hidden_d, cell_d) 
                out_d, hidden_d, cell_d = self.decoder(input_d, hidden_d, cell_d) 
                outputs[t, :, i] = out_d.squeeze() # out_d.shape was = (N, E=1)
                input_d = out_d.unsqueeze(0) # input_d.shape = (1, N, 1)
        
        return(outputs)
    
    def save(self, path='last_model'):
        "Saves model to the saved_models folder"
        torch.save(self.state_dict(), path)
        #print(f'Saved model to {path}')
        return None
