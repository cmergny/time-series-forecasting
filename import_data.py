### IMPORTS

import torch
import numpy as np
import pickle
from  torch.utils.data import Dataset
import matplotlib.pyplot as plt


### CLASSES

class Data:
    """Contains imported or generated multivariate time series
       Attributes structures the data for ML training"""
       
    def __init__(self, modes=range(10), nbr_snaps=-1, filename=None, multivar=False) -> None:
        self.x_train, self.y_train = [], []
        self.x_valid, self.y_valid = [], []
        self.multivar = multivar
        self.modes = modes
        # Generate
        if filename == None:
            self.data = self.GenerateData(modes=modes)
            self.filename = 'generated_sinus'
        # or Import
        else:
            self.filename = filename
            self.data = self.ImportData(filename, modes, nbr_snaps)
        self.Quantization(round=4)
        
    ## INITIALASING THE DATA
    
    def ImportData(self, filename, modes=range(10), nbr_snaps=-1):
        """Open and read coeff file. Truncate and normalize"""
        # Import
        if filename[-4:] == ".dat":
            with open("Data/podcoeff_095a05.dat", "rb") as f:
                egeinvalue = pickle.load(f)
                data = np.array(pickle.load(f))      
        elif filename[-7:] == ".pickle":
            with open(filename, "rb") as handle:
                eigd, eigvec, meanfield, x, y, z = pickle.load(handle)
                data = np.array(eigvec)
        # Pod from Yann
        elif filename[-2:] == ".d":
            data = np.loadtxt(filename)[:, 1:]
            
        elif filename == "data/spring_data.txt":
            data = np.loadtxt(filename)[100:, :] 
        # Pod from Berangere
        else:
            data = np.loadtxt(filename).reshape(305, 305, 3)  # Time, Mode, an(t)
            data = data[:, :, 2]
            data = np.transpose(data)
            
        # Truncate and Normalise
        data = data[:nbr_snaps, modes]
        print(f'Using data from {filename}')
        return(self.Normalise(data))
    
    def GenerateData(self, tf=2*np.pi, nbr_points=2000, modes=range(10)):
        """Generate artificial data with sinusoidal shape"""
        t = np.linspace(0.0, tf*6, nbr_points) # time array
        data = np.zeros((t.size, len(modes)))
        # Amplitude modulation
        for idx, i in enumerate(modes):
            f = float((i+1)* 0.1)
            f_m = f / 2
            A = 1
            B = 0.0
            ct = A * np.cos(2 * np.pi * f * t)
            mt = B * np.cos(2 * np.pi * f_m * t + np.random.rand())
            data[:, idx] = (1 + mt / A) * ct
        print('Using artificialy generated sinusoidal dataset.')
        # Return Nomalised dataset
        return(self.Normalise(data))
    
    def Normalise(self, data):
        """Normalise data for training"""
        for i in range(data.shape[1]):
            data[:, i] -= np.mean(data[:, i])  # remove mean value
            # normalize
            max = np.max(np.abs(data[:, i]))
            if max > 0: # make sure no div by 0
                data[:, i] /= max   
        return data

    def Quantization(self, round=1e3):
        """Rounds the continous value in the data by quantas"""
        self.data = np.round(self.data, round)
        
    ## PREPARING THE DATA
    
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
        # Add Noise for no overfit
        x_train = x_train + np.random.normal(0, 0.02, x_train.shape) if noise else x_train
        # reshape
        if not self.multivar:
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
        """Subsamples time serie into an array X of multiple windows of size iw, 
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
        ax.set_ylabel('amplitude')
        ax.set_xlabel('timesteps')
        plt.title('The first and last elements of the dataset')
        plt.savefig(path+'data_exemples')
           
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
    print(mydata)
