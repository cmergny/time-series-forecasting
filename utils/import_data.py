
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:28:56 2021
@author: Cyril
"""

import torch
import numpy as np
import pickle
from  torch.utils.data import Dataset
import matplotlib.pyplot as plt


### DATA

class Data:
    
    def __init__(self, modes, nbr_snaps=-1, file_name=None) -> None:
        
        self.x_train, self.y_train = [], []
        self.x_valid, self.y_valid = [], []
        # Generate
        if file_name == None:
            self.data = self.GenerateData()
        # or Import
        else:
            self.file_name = file_name
            self.modes = modes
            self.data = self.ImportData(file_name, modes, nbr_snaps)
        self.Quantization(round=1)
        
    ### INITIALASING THE DATA
    
    def ImportData(self, file_name, modes=range(10), nbr_snaps=-1):
        """Open and read coeff file. Truncate and normalize"""
        # Import
        if file_name[-4:] == ".dat":
            with open("Data/podcoeff_095a05.dat", "rb") as f:
                egeinvalue = pickle.load(f)
                data = np.array(pickle.load(f))      
        elif file_name[-7:] == ".pickle":
            with open(file_name, "rb") as handle:
                eigd, eigvec, meanfield, x, y, z = pickle.load(handle)
                data = np.array(eigvec)
        # Pod from Yann
        elif file_name[-2:] == ".d":
            data = np.loadtxt(file_name)[:, 1:]
        # Pod from Berangere
        else:
            data = np.loadtxt(file_name).reshape(305, 305, 3)  # Time, Mode, an(t)
            data = data[:, :, 2]
            data = np.transpose(data)
            
        # Truncate and Normalise
        data = data[:nbr_snaps, modes]
        print(f'Imported {file_name}')
        return(self.Normalise(data))
    
    def GenerateData(self, tf=2*np.pi, length=400, nbr_features=10):
        """Generate artificial data with sinusoidal shape"""
        t = np.linspace(0.0, tf, length) # time array
        data = np.zeros((t.size, nbr_features))
        # Amplitude modulation
        for i in range(nbr_features):
            f = float(i * 0.5 + 10)
            f_m = f / 2
            A = 1
            B = 0.0
            ct = A * np.cos(2 * np.pi * f * t)
            mt = B * np.cos(2 * np.pi * f_m * t + np.random.rand())
            data[:, i] = (1 + mt / A) * ct
        print('Generated artificial sinusoidal dataset')
        # Return Nomalised dataset
        return(self.Normalise(data))
    
    def Normalise(self, data):
        for i in range(data.shape[1]):
            data[:, i] -= np.mean(data[:, i])  # remove mean value
            data[:, i] /= np.max(np.abs(data[:, i]))  # normalize
        return data

    def Quantization(self, round=1e3):
        self.data = np.round(self.data, round)
        
    ### PREPARING THE DATA
    
    def PrepareDataset(self, split=0.7, noise=None, in_out_stride=(100, 30, 10), device=None):
        """Split dataset into four torch tensors x_train, x_valid, y_train, y_valid"""
        # Nested Function
        def Convert2Torch(*args, device):
            """Convert numpy array to torch tensor with device cpu or gpu."""
            return [torch.from_numpy(arg).float().to(device) for arg in args]
        # Split
        self.split = split
        self.data_train, self.data_test = self.SplitDataset()
        # Generate Inputs and Targets
        self.iw, self.ow, self.stride = in_out_stride  # input window, output window, stride
        x_train, y_train = self.WindowedDataset(self.data_train)
        x_valid, y_valid = self.WindowedDataset(self.data_test)
        # Noise
        x_train = x_train + np.random.normal(0, 0.02, x_train.shape) if noise else x_train
        # reshape
        Reshaping = lambda x: x.reshape(-1, x.shape[1]*x.shape[2], 1)
        x_train = Reshaping(x_train)
        y_train = Reshaping(y_train)
        x_valid = Reshaping(x_valid)
        y_valid = Reshaping(y_valid)
        # Convert tensor and set device
        self.device = device if device is not None else torch.device("cuda")  # train on cpu or gpu
        self.x_train, self.y_train, self.x_valid, self.y_valid =  Convert2Torch(x_train, y_train, x_valid, y_valid, device=self.device)
        self.train_ds = CustomDataset(self.x_train, self.y_train)
        self.valid_ds = CustomDataset(self.x_valid, self.y_valid)
        
    def SplitDataset(self, common=0.2):
        """
        Splits dataset into training and testing sets.
        split [float] : 0 < split <1, pourcentage of data to go in training set
        """
        idx = int(self.split * len(self.data))
        common = int(common * len(self.data))
        data_train = self.data[:idx, :]  # Add one dimension to array
        data_test = self.data[idx - common :, :]
        return(data_train, data_test)
    
    def WindowedDataset(self, data_group):
        """ Subsamples time serie into an array X of multiple windows of size iw, 
        and an array Y including target windows of size ow.
        iw [int]     : number of y samples to give model
        ow [int]     : number of future y samples to predict
        stride [int] : spacing between windows
        nbr_features [int] : number of features (i.e., 1 for us, but we could have multiple features)
        X, Y [np.array] : arrays with correct dimensions for LSTM (input/output window size, # of samples, # features])
        """
        # Compute how much samples required
        nbr_samples = (data_group.shape[0] - self.iw - self.ow) // self.stride + 1
        # Initialise Input and Target vectors
        nbr_features = data_group.shape[1]
        x = np.zeros([self.iw, nbr_samples, nbr_features])  # Input vector
        y = np.zeros([self.ow, nbr_samples, nbr_features])  # Target/Label vector
        # Iterate through multivariables
        for j in range(nbr_features):
            # Iterate through samples
            for i in range(nbr_samples):
                start = self.stride * i
                end = start + self.iw
                # Build Train
                x[:, i, j] = data_group[start:end, j]
                # Build Target
                y[:, i, j] = data_group[end-1 : end-1+self.ow, j]
        return(x, y)
    
    def __repr__(self) -> str:
        text =  f'device : {self.device}\n'
        text += f'x_train : {self.x_train.shape}\n'
        text += f'y_train : {self.y_train.shape}\n'
        text += f'x_valid : {self.x_valid.shape}\n'
        text += f'y_valid : {self.y_valid.shape}\n'
        
        plt.plot(self.x_train[:, 0, 0].to('cpu').detach())
        plt.plot(self.x_train[:, -1, 0].to('cpu').detach())
        return(text)
        
           
class CustomDataset(Dataset):
    "Used in the data class"
    def __init__(self, x, y) -> None:
        self.x = x # (S, N, E)
        self.y = y # (T, N, E)
        
    def __len__(self) -> int:
        return(self.x.shape[1])
    
    def __getitem__(self, index: int):
        return(self.x[:, index, :], self.y[:, index, :])
    

#%% MAIN
if __name__ == "__main__":
    mydata = Data()
    mydata.PrepareDataset(device="cpu")

