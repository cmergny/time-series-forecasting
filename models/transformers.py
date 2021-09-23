### IMPORTS

import math
import torch
import torch.nn as nn

### CLASSES
    
class Transformer(nn.Module):
    """ Tranformer Neural Network based on "Attention is all you need"
    paper and Neo Wu et al. 2020 (Transformer Models for Time series Forecasting."""
    
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.name = 'Transformer'
        # Layers
        self.linear1 = nn.Linear(in_features=1, out_features=d_model) # always applies on last dim
        self.posencoding = PositionalEncoding(d_model=d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=d_model*2, dropout=0.2)
        self.linear2 = nn.Linear(in_features=d_model, out_features=1)
        
    def forward(self, src, tgt):
        """ For transformers, the input x needs to be split 
        into a source and target. See shapes below."""
        # Expressing vectors to higher dims
        src = self.linear1(src) # (S, N, E) 
        tgt = self.linear1(tgt) # (T, N, E) 
        # Generate Masks
        src_mask = self.transformer.generate_square_subsequent_mask(src.shape[0]).to(src.device) # (S, S)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(tgt.device) # (T, T)
        # Pos Encoding
        src = self.posencoding(src)
        tgt = self.posencoding(tgt)
        # Transfo
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        # Linear
        out = self.linear2(out)
        return(out)

class PositionalEncoding(nn.Module):
    """Encodes position into a tensor from Attention is all you need paper"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]*0.1
        return self.dropout(x)

### MAIN
if __name__ == '__main__':
    print('This module is not a main script.')