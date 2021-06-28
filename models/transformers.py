### IMPORTS

import math
import torch
import torch.nn as nn
#from soft_dtw_cuda import SoftDTW

### CLASSES



class MyModel(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        # Layers
        self.linear1 = nn.Linear(in_features=1, out_features=d_model) # always third dim modified
        self.posencoding = PositionalEncoding(d_model=d_model)
        self.transformer = MyTransformer(d_model=d_model, nhead=8, num_encoder_layers=6, dropout=0.1)
        self.linear2 = nn.Linear(in_features=d_model, out_features=1)
        
    def forward(self, src):
        # Expressing vectors to higher dims
        src = self.linear1(src) # (S, N, E) 
        src_mask = self.transformer.generate_square_subsequent_mask(src.shape[0]).to(src.device) # (S, S)
        # Pos Encoding
        src = self.posencoding(src)
        # Transfo
        out = self.transformer(src, src_mask=src_mask)
        # Linear
        out = self.linear2(out)
        return(out)

class MyTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)        
        self._reset_parameters()

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        output = self.transformer_encoder(src, src_mask)
        return output
    

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
    
class RealTransfo(nn.Module):

    def __init__(self, d_model, nhead=8):
        super().__init__()
        # Layers
        self.linear1 = nn.Linear(in_features=1, out_features=d_model) # always third dim modified
        self.posencoding = PositionalEncoding(d_model=d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=d_model*2, dropout=0.2)
        self.linear2 = nn.Linear(in_features=d_model, out_features=1)
        
    def forward(self, src, tgt):
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

#%% MAIN
if __name__ == "__main__":
    pass