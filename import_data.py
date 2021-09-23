# Module used for importing arrays of data
# in the multiples files found in the dir data/
# Imports may differ accoring to the file, feel free
# to customize it.
# Splits and convert data for pytorch neural network
# training.

### IMPORTS

import torch
import numpy as np
import pickle
from  torch.utils.data import Dataset
import matplotlib.pyplot as plt


### CLASSES

class Data:
    """Contains imported or generated multivariate time series.
    Data files must have 2 columns. (nbr of time steps, nbr of variables)
    For training, each mode is cut into multiple sub windows.
    x_train.shape = (S, N, E) 
    y.train.shape = (T, N, E)
    with S and T the (resp) source and target length, N the number of subwindows for 
    each mode, and E the total number of modes (or variables).
    """
       
    def __init__(self, modes=range(10), final_time=-1, filename=None, multivar=True) -> None:
        self.multivar = multivar
        self.modes = modes
        self.filename = filename
        self.data = self.import_data(filename, modes, final_time)
        self.data = np.round(self.data, 4) # Discretizes the data
        self.data = self.data[:final_time, modes] # Truncation
        self.data = self.normalise(self.data)
        
    def import_data(self, filename, modes=range(10), final_time=-1):
        """Import data array of 2 columns: (timesteps)| (modes)
        Don't mind
        """
        # Import varies accoring to input file
        if filename[-4:] == ".dat":
            with open("Data/podcoeff_095a05.dat", "rb") as f:
                egeinvalue = pickle.load(f)
                data = np.array(pickle.load(f))    
        # Other Pod Berange    
        elif filename[-7:] == ".pickle":
            with open(filename, "rb") as handle:
                eigd, eigvec, meanfield, x, y, z = pickle.load(handle)
                data = np.array(eigvec)
        # Pod from Yann
        elif filename[-2:] == ".d":
            data = np.loadtxt(filename)[:, 1:]
        # Spring simualtions
        elif filename == "data/spring_data.txt":
            data = np.loadtxt(filename)[100:, :] 
        # Pod from Berangere
        elif filename == "coeff":
            data = np.loadtxt(filename).reshape(305, 305, 3)  # Time, Mode, an(t)
            data = np.transpose(data[:, :, 2])
        else:
            data = np.loadtxt(filename)
        print(f'Using data from {filename}')
        return(data)

    def normalise(self, data):
        """normalise data if necessary."""
        for i in range(data.shape[1]):
            data[:, i] -= np.mean(data[:, i])  # remove mean value
            # normalize
            max = np.max(np.abs(data[:, i]))
            if max > 0: # make sure no div by 0
                data[:, i] /= max   
        return data        
        
    ## PREPARING THE DATA
    
    def prepare_dataset(self, split=0.7, noise=None, in_out_stride=(100, 30, 10)):
        """Split dataset into four torch tensors x_train, x_valid, y_train, y_valid"""
        # Nested Function
        def Convert2Torch(*args, device):
            """Convert numpy array to torch tensor with device cpu or gpu."""
            return [torch.from_numpy(arg).float().to(device) for arg in args]
        # Split into training and validation set
        self.data_train, self.data_test = self.split_dataset(split)
        # Generate Inputs and Targets
        self.iw, self.ow, self.stride = in_out_stride  # input window, output window, stride
        x_train, y_train = self.windowed_data(self.data_train)
        x_valid, y_valid = self.windowed_data(self.data_test)
        # Add Noise for no overfit
        x_train = x_train + np.random.normal(0, 0.05, x_train.shape) if noise else x_train
        x_valid = x_valid + np.random.normal(0, 0.05, x_valid.shape) if noise else x_valid
        # Reshape into a (S, N*E, 1) array if not multivar
        if not self.multivar:
            Reshaping = lambda x: x.reshape(-1, x.shape[1]*x.shape[2], 1)
            x_train, y_train, x_valid, y_valid = [Reshaping(a) for a in [ x_train, y_train, x_valid, y_valid]]
        # Convert tensor and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # train on cpu or gpu
        self.x_train, self.y_train, self.x_valid, self.y_valid =  Convert2Torch(x_train, y_train, x_valid, y_valid, device=self.device)
        print(self)
        
    def split_dataset(self, split, common=0.0):
        """
        Splits dataset into training and testing sets.
        split [float] : 0 < split <1, pourcentage of data to go in training set
        """
        idx = int(split * len(self.data))
        common = int(common * len(self.data))
        data_train = self.data[:idx, :]  # Add one dimension to array
        data_test = self.data[idx - common :, :]
        return(data_train, data_test)
    
    def windowed_data(self, data_group):
        """Subsamples time serie into an array X of multiple windows of size iw, 
        and an array Y including target windows of size ow.
        iw [int]     : number of y samples to give model
        ow [int]     : number of future y samples to predict
        stride [int] : spacing between windows
        X, Y [np.array] : arrays with correct dimensions for LSTM 
            (input/output window size, # of samples, # features])
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
        """Printing an instance of the Data class shows its
           important attributes"""
        text = f'\nData : \n\tfile name : {self.filename}\n'
        text += f'\tmodes selected : {self.modes[0]} to {self.modes[-1]}\n'
        text += f'\tin, out, stride : {self.iw}, {self.ow}, {self.stride}\n'
        text += f'\tx_train : {self.x_train.shape}\n'
        text += f'\ty_train : {self.y_train.shape}\n'
        text += f'\tx_valid : {self.x_valid.shape}\n'
        text += f'\ty_valid : {self.y_valid.shape}\n'
        return(text)
    
    def plot(self, path):
        """Plots last and first element of data"""
        if self.multivar:
            x_first = self.x_train[:, 0, 0].to('cpu').detach()
            x_last = self.x_train[:, 0, -1].to('cpu').detach()
        else:
            x_first = self.x_train[:, 0, 0].to('cpu').detach()
            x_last = self.x_train[:, -1, 0].to('cpu').detach()
        fig, ax = plt.subplots()
        ax.plot(x_first)
        ax.plot(x_last)
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Timesteps')
        plt.title('The first and last elements of the dataset')
        plt.savefig(path+'data_exemples')
           
           
### MAIN
if __name__ == '__main__':
    print('This module is not a main script.')
