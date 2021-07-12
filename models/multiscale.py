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

class MultiScaleAttention(nn.Module):
    
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, C, hidden_d):
        # C.shape = (E, N, H)
        hidden_d = hidden_d.repeat(C.shape[0], 1, 1) 
        energy = self.linear3(self.tanh(self.linear1(C)+self.linear2(hidden_d))) # (E, N, 1)
        beta = self.softmax(energy) # (E, N, 1)
        c = torch.einsum("enl,enh->lnh", beta, C) # c.shape = (1, N, H)
        return(c, beta)      
    
class MultiScaleLSTMA(nn.Module):
    
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoders =  nn.ModuleList([Encoder(1, hidden_size) for _ in range(input_size)])
        self.decoders = nn.ModuleList([Decoder(1, hidden_size) for _ in range(input_size)])
        self.multiscale_attention = MultiScaleAttention(hidden_size)
        
    def forward(self, x, target_len):       
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
        
        # Call Decoder for all modes
        for i in range(self.input_size):
            input_d = x[-1, :, i].unsqueeze(-1).unsqueeze(0)
            hidden_d, cell_d = self.hiddens[i].unsqueeze(0), self.cells[i].unsqueeze(0)
            hidden_d, alpha = self.multiscale_attention(self.hiddens, hidden_d)
            # issue All modes give the same attention !
            self.alphas[:, :, i] = alpha.squeeze(-1) # alpha.shape = (E, N, 1) -> (E, N)
            
            for t in range(target_len):
                out_d, hidden_d, cell_d = self.decoders[i](input_d, hidden_d, cell_d)
                # out.shape = (N, E=1)
                outputs[t, :, i] = out_d.squeeze() 
                input_d = out_d.unsqueeze(0) # input_d.shape = (1, N, 1)
                
        return(outputs)
    
    def save(self, path='last_model'):
        "Saves model to the saved_models folder"
        torch.save(self.state_dict(), path)
        #print(f'Saved model to {path}')
        return None
