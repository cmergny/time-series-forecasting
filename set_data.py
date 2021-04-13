
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:28:56 2021
@author: Cyril
"""

import torch
import numpy as np

def ReadCoeff(file_name = 'coeff'):
    """ Open and read coeff file
        returns an array of length (# of modes, # of snapshots, 3) """    
    data = np.loadtxt(file_name).reshape(305, 305, 3) # Time, Mode, an(t) 
    return(data)

def TruncateNormData(ar, modes=[1, 2, 3, 4], nbr_snaps=300, index=2):
    """Truncate the input data according to values of interest
    and normalize"""
    X = ar[modes, :nbr_snaps, index]
    X = np.transpose(X)
    for i in range(len(modes)):
        X[:, i] -= np.mean(X[:, i]) # remove mean value
        X[:, i] /= np.max(np.abs(X[:, i])) # normalize
    return(X)

def Convert2Torch(*args, device):
    """Convert numpy array to torch tensor with device cpu or gpu."""
    return([torch.from_numpy(arg).float().to(device) for arg in args])

def MakeDataset(Data, split=0.8, common=0.2):
    """
    Splits dataset into training and testing sets.
    split [float] : 0 < split <1, pourcentage of data to go in training set
    """
    idx = int(split*len(Data))
    common = int(common*len(Data))
    Data_train = Data[:idx, :] # Add one dimension to array
    Data_test = Data[idx-common:, :]  
    return(Data_train, Data_test)

def WindowedDataset(Data, iw=5, ow=1, stride=1, nbr_features=1):
    """
    Subsamples time serie into an array X of multiple windows of size iw, 
    and an array Y including target windows of size ow.
    ----------
    Data [int]   : time series feature (array)
    iw [int]     : number of y samples to give model 
    ow [int]     : number of future y samples to predict  
    stride [int] : spacing between windows   
    nbr_features [int] : number of features (i.e., 1 for us, but we could have multiple features)
    ----------
    X, Y [np.array] : arrays with correct dimensions for LSTM (input/output window size, # of samples, # features])
    """
    # Compute how much samples required
    nbr_samples = (Data.shape[0] - iw - ow) // stride + 1
    # Initialise Input and Target vectors
    X = np.zeros([iw, nbr_samples, nbr_features])  # Input vector 
    Y = np.zeros([ow, nbr_samples, nbr_features]) # Target/Label vector
    
    # Iterate through multivariables
    for j in range(nbr_features):
        # Iterate through samples
        for i in range(nbr_samples):
            start = stride*i
            end = start + iw
            # Build Train
            X[:, i, j] = Data[start:end, j]
            # Build Target
            Y[:, i, j] = Data[end:end+ow, j]
            
    return X, Y