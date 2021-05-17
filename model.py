# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:37:09 2021

@author: Cyril
"""

### IMPORTS

import numpy as np
from tqdm import trange
from soft_dtw_cuda import SoftDTW
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch import optim

### CLASSES

class Encoder(nn.Module):
    """ Used to encode a sequence of time series """
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        """ Initialising var and defining LSTM 
        shape: nn.LSTM(input_size, hidden_size, numlayers)"""
        super().__init__()
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
    def forward(self, input_batch):
        """
        input_batch: input of shape (seq_len, nbr in batch, input_size)
        --------
        lstm_out: gives all the hidden states in the sequence
        hidden: gives the hidden state & cell state of the last elt of the seq
        ----
        lstm_out.shape = (seqlen, bs, hidden_size) = h for all seq len
        """  
        lstm_out, self.hidden = self.lstm(input_batch) # hidden not provided : both h_0 and c_0 default to zero.
        return(lstm_out, self.hidden)

class Decoder(nn.Module):
    """ Use hidden state init by Encoder to make predictions """
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        """ Initialising var and defining LSTM 
        shape: nn.LSTM(input_size, hidden_size, numlayers)"""
        super().__init__()  
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) #initalize
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, input_batch, encoder_hidden_states):
        """
        input_batch: input of shape (seqlen=1, bs, hidden_size)
        encoder_hidden_states: tuple (h, c) 
        --------
        lstm_out.shape = (seqlen=1, bs, hiddensize) = h for all seq len
        self.hidden.shape = (h, c)
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
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size)

    def forward(self, input_batch, target_len):
        # Initialise outputs (targetlen, bs, # features)
        outputs = torch.zeros(target_len,  input_batch.shape[1], input_batch.shape[2]).to(input_batch.device)
        # Initialise h,c and call Encoder 
        encoder_output, encoder_hidden = self.encoder(input_batch)
        # Initialise Decoder
        decoder_input = input_batch[-1, :, :].unsqueeze(0) # shape(bs, n_features)
        decoder_hidden = encoder_hidden
        # Iterate by values to predict
        for t in range(target_len):
            # Call Decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output.unsqueeze(0)
        return(outputs)

### FUNCTIONS

def TrainModel(model, input_tensor, target_tensor, x_valid, y_valid, n_epochs, target_len,
            batch_size, lr, wd):
    """
    Train the LSTM Encoder-Decoder NN
    --------
    input_tensor [Pytorch tensor] : input data of shape (seq_len, nbr in batch, input_size)
    target_tensor [Pytorch tensor] : target data of shape (seq_len, nbr in batch, input_size)    
    n_epochs [int] : nbr of epochs
    target_len [int] : nbr of values to predict
    batch_size [int] : nbr of samples per gradient update
    lr [float] : learning rate to update weights, lr >= 0
    --------
    losses: array of loss function for each epoch
    """
    # Loss and optimizer
    writer = SummaryWriter()
    Losses = np.full(n_epochs, np.nan) # Init losses array with NaNs
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    #criterion = SoftDTW(use_cuda=True, gamma=0.1)
    nbr_batches = int(input_tensor.shape[1] / batch_size) # nbr of batch iterations
    # Iterate by epochs
    with trange(n_epochs) as tr:
        for ep in tr:
            # training loop
            model.train()
            batch_loss = 0
            # Iterate by batch
            for b in range(nbr_batches):
                # Select batches
                input_batch = input_tensor[:, b:b+batch_size, :]
                target_batch = target_tensor[:, b:b+batch_size, :]
                # Initialise gradient to zero
                optimizer.zero_grad()
                # Calling model
                outputs = model(input_batch, target_len)
                # Computing loss
                loss = criterion(outputs, target_batch)
                batch_loss += loss
                # Backpropagating 
                loss.backward()
                optimizer.step()
            # Computing Loss FOR epoch
            batch_loss /= nbr_batches 
            writer.add_scalar("Train_Loss", batch_loss, ep)
            Losses[ep] = batch_loss

            # evaluation loop
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                nbr_batches_valid = int(x_valid.shape[1]/batch_size) 
                for b in range(nbr_batches_valid):
                    input_batch = x_valid[:, b:b+batch_size, :]
                    target_batch = y_valid[:, b:b+batch_size :]
                    outputs = model(input_batch, target_len)
                    loss = criterion(outputs, target_batch)
                    valid_loss += loss
                valid_loss = valid_loss/nbr_batches_valid
                writer.add_scalar("Valid_Loss", valid_loss, ep)
            model.train()

            tr.set_postfix(train="{0:.2e}".format(batch_loss), valid="{0:.2e}".format(valid_loss)) 
    writer.flush()        
    return(Losses)
                    

def Predict(model, input_batch, target_len):
    model.eval()
    with torch.no_grad():
        outputs = model(input_batch[:, :, :], target_len)
        return(outputs.detach())
