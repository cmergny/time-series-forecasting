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
        
    def forward(self, x_input):
        """
        x_input: input of shape (seq_len, nbr in batch, input_size)
        --------
        lstm_out: gives all the hidden states in the sequence
        hidden: gives the hidden state & cell state of the last elt of the seq
        """  
        lstm_out, self.hidden = self.lstm(
            x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
    
        return(lstm_out, self.hidden)
        
    def init_hidden(self, batch_size):
        """
        batch_size: x_input.shape[1]
        --------
        return:  hidden state and cell state initialised with zeros
        """
        return(torch.zeros(self.num_layers, batch_size, self.hidden_size),
               torch.zeros(self.num_layers, batch_size, self.hidden_size))


class Decoder(nn.Module):
    """ Decodes hidden state output by encoder """
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        """ Initialising var and defining LSTM """
        super().__init__()
        self.input_size = input_size   # nbr of features in input X
        self.hidden_size = hidden_size # size of the hidden state
        self.num_layers = num_layers   # nbr of recurrent layers
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) #initalize
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, x_input, encoder_hidden_states):
        """
        x_input: should be 2D (batch_size, input_size)
        encoder_hidden_states: hidden states
        --------
        output: gives all the hidden states in the sequence
        hidden: gives the hidden state & cell state of the last elt of the seq
        """  
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
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
        
    def train_model(self, input_tensor, target_tensor, input_test, target_test, n_epochs, target_len,
                    batch_size, learning_rate, wd, device):
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
        Losses = np.full(n_epochs, np.nan) # Init losses array with NaNs
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=wd)
        criterion = nn.MSELoss()
        #criterion = SoftDTW(use_cuda=True, gamma=0.1)
        nbr_batches = int(input_tensor.shape[1] / batch_size) # nbr of batch iterations
                
        with trange(n_epochs) as tr:
            # Iterate by epochs
            for ep in tr:
                batch_loss = 0
                # Iterate by batch
                for b in range(nbr_batches):
                    # Select batches
                    input_batch = input_tensor[:, b:b+batch_size, :]
                    target_batch = target_tensor[:, b:b+batch_size, :]
                    # Initialise outputs
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2]).to(device)
                    # Initialise hidden states h and c 
                    encoder_hidden = self.encoder.init_hidden(batch_size)
                    # Initialise gradient to zero
                    optimizer.zero_grad()
                    # Call Encoder
                    encoder_output, encoder_hidden = self.encoder(input_batch)
                    # Initialise Decoder
                    decoder_input = input_batch[-1, :, :] # shape (batch_size, input_size)
                    decoder_hidden = encoder_hidden
                    #print(encoder_hidden.shape)
                    
                    # Iterate by values to predict
                    for t in range(target_len):
                        # Call Decoder
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = decoder_output
                      
                    # Computing loss
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.mean()
                    # Backpropagating 
                    loss.mean().backward()
                    optimizer.step()
                    
                # Computing Loss FOR epoch
                batch_loss /= nbr_batches 
                Losses[ep] = batch_loss

                # Test loss
                #i = 1
                #output_test = self.predict(input_test[:,i,:], target_len)
                #output_test = torch.from_numpy(output_test).float().to(device)
                #test_loss = criterion(output_test[:,:], target_test[:,i,:])
                # Progress bar
                tr.set_postfix(loss = "{0:.2e}".format(batch_loss)) #, test_loss = "{0:.2e}".format(test_loss))
                
        return(Losses)
                    
    def predict(self, input_tensor, target_len):
        """
        Predict values with the LSTM Encoder-Decoder NN
        --------
        input_tensor [Pytorch tensor] : 2D input data of shape (seq_len, input_size)
        target_len [int] : number of target values to predict
        --------
        outputs [np.array]: array containing predicted values 
        """
        # Encode input tensor by calling Encoder
        input_tensor = input_tensor.unsqueeze(1) # add in a batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)
        
        # Initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])
        # Initialize decoder inputs
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        # Calling Decoder
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
        outputs = np.array(outputs.detach())
        return(outputs)
        
         