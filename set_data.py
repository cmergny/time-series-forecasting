
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:28:56 2021
@author: Cyril
"""

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def ImportData(file_name='Data/coeff', modes=range(10), nbr_snaps=300, index=2):
    """ Open and read coeff file
        returns an array of length (# of modes, # of snapshots, 3)
        Truncate and normalize """
    if file_name[-4:] == '.dat': 
        with open("Data/podcoeff_095a05.dat", "rb") as f:
            egeinvalue = pickle.load(f)
            data = np.array(pickle.load(f))  # (seqlen, modes)
            data = data[:nbr_snaps, modes]
    else: # if file from 3D POD
        data = np.loadtxt(file_name).reshape(305, 305, 3) # Time, Mode, an(t) 
        data = data[modes, :nbr_snaps, index]
        data = np.transpose(data)
    # Normalise
    for i in range(len(modes)):
        data[:, i] -= np.mean(data[:, i])
        data[:, i] /= np.max(np.abs(data[:, i])) 
    return(data)

def Convert2Torch(*args, device):
    """Convert numpy array to torch tensor with device cpu or gpu."""
    return([torch.from_numpy(arg).float().to(device) for arg in args])

def SplitDataset(Data, split=0.8, common=0.2):
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

def GenerateData(tf=2*np.pi, L=1000, freq=[1]):
    t = np.linspace(0., tf, L)
    data = np.zeros((t.size, len(freq)))
    for i, f in enumerate(freq):
        data[:, i] = np.cos(f*t)
        data[:, i] -= np.mean(data[:, i]) # remove mean value
        data[:, i] /= np.max(np.abs(data[:, i])) # normalize
    return(data)

def PrepareDataset(data, split=0.7, noise=None, in_out_stride=(100, 30, 10)):
    """ Returns 4 torch tensors"""
    # Split
    data_train, data_test = SplitDataset(data, split=split)
    # Generate Inputs and Targets
    iw, ow, stride = in_out_stride # input window, output window, stride
    x_train, y_train = WindowedDataset(data_train, iw, ow, stride, nbr_features=data_train.shape[1]) 
    x_valid, y_valid = WindowedDataset(data_test, iw, ow, stride, nbr_features=data_test.shape[1])
    # Noise
    x_train = x_train + np.random.normal(0, 0.02, x_train.shape) if noise else x_train
    # Convert tensor and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # train on cpu or gpu
    return(Convert2Torch(x_train, y_train, x_valid, y_valid, device=device))


def PCA(data, thresh=0.5):
    """
    Return estimation of dataset intrasic dimension using
    Principal Component Analysis
    """
    R = np.corrcoef(data.transpose())
    values, vectors = np.linalg.eig(R)
    #values = values/np.max(values)
    print('id = ', len(values[values>thresh]))
    return(values)

def MaxLikelihood(data, k):
    """
    Return estimation of dataset intrasic dimension using
    article: Elizaveta Levina, Peter J. Bickel
    Maximum Likelihood Estimation of Intrinsic Dimension"""
    n = data.shape[1]
    Matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Eucl dist
            Matrix[i,j] = np.linalg.norm(data[:, i]-data[:, j])
    Matrix.sort() # sort by NN
    # k is nbr of neirest neigbhors
    mk = 0 # computed dimension
    for x in range(n):
        mkx = 0
        for j in range(1, k): # to k-1  
            mkx += np.log(Matrix[x, k]/Matrix[x, j])
        mk +=  (1/(k-1)*mkx)**(-1)
    mk = mk/n
    return(mk)

def AirQualityData(width=500):
    """ import the air qualityfile """
    data = np.zeros((width, 13-2))
    df = pd.read_csv('Data/air_quality.csv', sep=';', index_col=False)
    # import wanted columns
    for j, column in enumerate(df.columns[2:13]):
        ar = np.array(df[column][:width])
        # convert str to float
        for i, elt in enumerate(ar):
            if type(elt) == str:
                ar[i] = float(elt.replace(',', '.'))
            ar[i] = 0 if ar[i] < -100 else ar[i]
        # Normalise
        data[:, j] = ar - np.mean(ar)
        data[:, j] /= np.max(np.abs(data[:, j])) 
    return(data)

### MAIN
if __name__ == '__main__':
    m = 1
    n = 300
    data = ImportData(file_name="Data/coeff",  modes=range(m, m+n), nbr_snaps=300)

    # values = PCA(data)


    

    transforms = np.zeros(data.shape)
    for i in range(data.shape[1]):
        transforms[:, i] = np.abs(np.fft.fft(data[:, i]))
    transforms = transforms[:transforms.shape[0]//2, :]
    #plt.plot(transforms)
    #values = PCA(transforms)

    ar =[]
    for k in range(2, 200, 10):
        mk = MaxLikelihood(transforms, k)
        ar.append(mk)
    plt.plot(ar)


    
