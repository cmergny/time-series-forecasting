# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:37:09 2021

@author: Cyril
"""

### IMPORTS

import numpy as np
from tqdm import trange
from soft_dtw_cuda import SoftDTW

import torch
import torch.nn as nn
from torch import optim

### CLASSES

class Encoder(nn.Module):
    """ Used to encode a sequence of time series """
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        """ Initialising var and defining LSTM """
        super().__init__()
        self.input_size = input_size   # nbr of features in input X
        self.hidden_size = hidden_size # size of the hidden state
        self.num_layers = num_layers   # nbr of recurrent layers
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
    def forward(self, input_batch):
        """
        input_batch: input of shape (seq_len, nbr in batch, input_size)
        --------
        lstm_out: gives all the hidden states in the sequence
        hidden: gives the hidden state & cell state of the last elt of the seq
        """  
        lstm_out, self.hidden = self.lstm(input_batch) # hidden not provided : both h_0 and c_0 default to zero.
        return(lstm_out, self.hidden)

class Decoder(nn.Module):
    """ Use hidden state init by Encoder to make predictions """
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        """ Initialising var and defining LSTM """
        super().__init__()
        self.input_size = input_size   # nbr of features in input X
        self.hidden_size = hidden_size # nbr of hidden state per cell
        self.num_layers = num_layers   # nbr of recurrent layers
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) #initalize
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, input_batch, encoder_hidden_states):
        """
        input_batch: input of shape (seq_len, nbr in batch, input_size)
        encoder_hidden_states: tuple (h, c) 
                where h.shape = (nbr_layer*nbrdirect, batch_size, hidden_size)
        --------
        output: gives all the hidden states in the sequence (seqlen, batch_size, hidden_size)
        hidden: gives the hidden state & cell state of the last elt of the seq
        """  
        lstm_out, self.hidden = self.lstm(input_batch, encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return(output, self.hidden)
      
class LSTM_EncoderDecoder(nn.Module):
    """
    Use the two Encoder and Decoder classes to train a LSTM neural network
    Can also make predictions once the NN is trained
    """
    def __init__(self, input_size, hidden_size):
        """ Initialising variables with param and Encoder/Decoder classes"""
        super().__init__()
        self.input_size = input_size   # nbr of features in input X
        self.hidden_size = hidden_size # size of the hidden state
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size)

    def forward(self, input_batch, target_len):
        # Initialise outputs (targetlen, bs, # features)
        batch_size = input_batch.shape[1]
        outputs = torch.zeros(target_len, batch_size, input_batch.shape[2]).to(input_batch.device)
        # Initialise h,c and call Encoder 
        encoder_output, encoder_hidden = self.encoder(input_batch)
        # Initialise Decoder
        decoder_input = input_batch[-1, :, :].unsqueeze(0) # shape (batch_size, input_size)
        decoder_hidden = encoder_hidden
        # Iterate by values to predict
        for t in range(target_len):
            # Call Decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output.unsqueeze(0)
        return(outputs)

### OTHER CLASS

class SimpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        """ Initialising var and defining LSTM """
        super().__init__()
        self.input_size = input_size   # nbr of features in input X
        self.hidden_size = hidden_size # size of the hidden state
        self.num_layers = num_layers   # nbr of recurrent layers
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) #initalize
        self.linear = nn.Linear(hidden_size, input_size)
                
    def forward(self, input_batch):
        """
        input_batch: input of shape (seq_len, nbr in batch, input_size)
        encoder_hidden_states: tuple (h, c) 
                where h.shape = (nbr_layer*nbrdirect, batch_size, hidden_size)
        --------
        output: gives all the hidden states in the sequence (seqlen, batch_size, hidden_size)
        hidden: gives the hidden state & cell state of the last elt of the seq
        """
        lstm_output, self.hidden_states = self.lstm(input_batch, self.hidden_states)
        lstm_output = self.linear(lstm_output.squeeze(0))
        return(lstm_output)
    
    def init_hidden(self, batch_size, device):
        self.hidden_states = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        return(self.hidden_states)

### FUNCTIONS

def TrainModel(model, input_tensor, target_tensor, n_epochs, target_len,
            batch_size, learning_rate, wd):
    """
    Train the LSTM Encoder-Decoder NN
    --------
    input_tensor [Pytorch tensor] : input data of shape (seq_len, nbr in batch, input_size)
    target_tensor [Pytorch tensor] : target data of shape (seq_len, nbr in batch, input_size)    
    n_epochs [int] : nbr of epochs
    target_len [int] : nbr of values to predict
    batch_size [int] : nbr of samples per gradient update
    learning_rate [float] : learning rate to update weights, lr >= 0
    --------
    losses: array of loss function for each epoch
    """
    # Loss and optimizer
    model.train()
    Losses = np.full(n_epochs, np.nan) # Init losses array with NaNs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    criterion = nn.MSELoss()
    #criterion = SoftDTW(use_cuda=True, gamma=0.1)
    nbr_batches = int(input_tensor.shape[1] / batch_size) # nbr of batch iterations
    # Iterate by epochs
    with trange(n_epochs) as tr:
        for ep in tr:
            batch_loss = 0
            # Iterate by batch
            for b in range(nbr_batches):
                model.init_hidden(batch_size, input_tensor.device)
                # Select batches
                input_batch = input_tensor[:, b:b+batch_size, :]
                target_batch = target_tensor[:, b:b+batch_size, :]
                # Initialise gradient to zero
                optimizer.zero_grad()
                # Calling model
                #outputs = model(input_batch, target_len)
                outputs = model(input_batch)[-1].unsqueeze(0)
                # Computing loss
                loss = criterion(outputs, target_batch)
                batch_loss += loss
                # Backpropagating 
                loss.backward()
                optimizer.step()
            # Computing Loss FOR epoch
            batch_loss /= nbr_batches 
            Losses[ep] = batch_loss
            # Progress bar
            tr.set_postfix(loss = "{0:.2e}".format(batch_loss))         
    return(Losses)
                    

def Predict(model, input_batch, target_len):
    model.eval()
    with torch.no_grad():
        model.init_hidden(input_batch.shape[1], input_batch.device)
        outputs = torch.zeros(target_len, input_batch.shape[1], input_batch.shape[2])
        for t in range(target_len):
            lstm_output = model(input_batch[:, :, :])
            outputs[t] = lstm_output[-1].unsqueeze(0)
            input_batch = lstm_output
        return(outputs.detach())
