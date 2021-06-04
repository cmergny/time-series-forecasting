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

### DATA


def Normalise(data):
    for i in range(data.shape[1]):
        data[:, i] -= np.mean(data[:, i])  # remove mean value
        data[:, i] /= np.max(np.abs(data[:, i]))  # normalize
    return data


def ImportData(file_name="Data/coeff", modes=range(10), nbr_snaps=300, index=2):
    """Open and read coeff file. Truncate and normalize
    podresults_ch180.pickle
    podresults_cavity.pickle
    """
    if file_name[-4:] == ".dat":
        with open("Data/podcoeff_095a05.dat", "rb") as f:
            egeinvalue = pickle.load(f)
            data = np.array(pickle.load(f))  # (seqlen, modes)
            data = data[:nbr_snaps, modes]
    if file_name[-7:] == ".pickle":
        with open(file_name, "rb") as handle:
            eigd, eigvec, meanfield, x, y, z = pickle.load(handle)
            data = np.array(eigvec)
            data = data[:nbr_snaps, modes]
    # Pod from Yann
    if file_name[-2:] == ".d":
        data = np.loadtxt(file_name)[:, 1:]
        data = data[:nbr_snaps, modes]
    # Pod from Berangere
    else:
        data = np.loadtxt(file_name).reshape(305, 305, 3)  # Time, Mode, an(t)
        data = data[modes, :nbr_snaps, index]
        data = np.transpose(data)
    return Normalise(data)


def SplitDataset(Data, split=0.8, common=0.2):
    """
    Splits dataset into training and testing sets.
    split [float] : 0 < split <1, pourcentage of data to go in training set
    """
    idx = int(split * len(Data))
    common = int(common * len(Data))
    Data_train = Data[:idx, :]  # Add one dimension to array
    Data_test = Data[idx - common :, :]
    return (Data_train, Data_test)


def WindowedDataset(Data, iw=5, ow=1, stride=1, nbr_features=1):
    """
    Subsamples time serie into an array X of multiple windows of size iw,
    and an array Y including target windows of size ow.
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
    Y = np.zeros([ow, nbr_samples, nbr_features])  # Target/Label vector

    # Iterate through multivariables
    for j in range(nbr_features):
        # Iterate through samples
        for i in range(nbr_samples):
            start = stride * i
            end = start + iw
            # Build Train
            X[:, i, j] = Data[start:end, j]
            # Build Target
            Y[:, i, j] = Data[end : end + ow, j]
    return X, Y


def PrepareDataset(data, split=0.7, noise=None, in_out_stride=(100, 30, 10)):
    """Split dataset into four torch tensors x_train, x_valid, y_train, y_valid"""

    def Reshaping(x):
        x = x.reshape(-1, x.shape[1] * x.shape[2], 1)
        return x

    def Convert2Torch(*args, device):
        """Convert numpy array to torch tensor with device cpu or gpu."""
        return [torch.from_numpy(arg).float().to(device) for arg in args]

    # Split
    data_train, data_test = SplitDataset(data, split=split)
    # Generate Inputs and Targets
    iw, ow, stride = in_out_stride  # input window, output window, stride
    x_train, y_train = WindowedDataset(data_train, iw, ow, stride, nbr_features=data_train.shape[1])
    x_valid, y_valid = WindowedDataset(data_test, iw, ow, stride, nbr_features=data_test.shape[1])
    # Noise
    x_train = x_train + np.random.normal(0, 0.02, x_train.shape) if noise else x_train
    # reshape
    x_train = Reshaping(x_train)
    y_train = Reshaping(y_train)
    x_valid = Reshaping(x_valid)
    y_valid = Reshaping(y_valid)
    # Convert tensor and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # train on cpu or gpu
    return Convert2Torch(x_train, y_train, x_valid, y_valid, device=device)


def GenerateData(tf=2 * np.pi, L=4000, nbr_samples=500):
    """Generate artificial data with sinusoidal shape"""
    t = np.linspace(0.0, tf, L)
    data = np.zeros((t.size, nbr_samples))
    # Amplitude modulation
    for i in range(nbr_samples):
        f = float(i * 0.01 + 1)
        f_m = f / 2
        A = 1
        B = 0.4
        ct = A * np.cos(2 * np.pi * f * t)
        mt = B * np.cos(2 * np.pi * f_m * t + np.random.rand())
        data[:, i] = (1 + mt / A) * ct
    # Return Nomalised dataset
    return Normalise(data)


def AirQualityData(width=500):
    """import the air quality file dataset"""
    data = np.zeros((width, 13 - 2))
    df = pd.read_csv("Data/air_quality.csv", sep=";", index_col=False)
    # import wanted columns
    for j, column in enumerate(df.columns[2:13]):
        ar = np.array(df[column][:width])
        # convert str to float
        for i, elt in enumerate(ar):
            if type(elt) == str:
                ar[i] = float(elt.replace(",", "."))
            ar[i] = 0 if ar[i] < -100 else ar[i]
    return Normalise(data)


### INTRASIC DIM ESTIMATORS


def PCA(data, thresh=0.5):
    """
    Return estimation of dataset intrasic dimension using
    Principal Component Analysis
    """
    R = np.corrcoef(data.transpose())
    values, vectors = np.linalg.eig(R)
    values = values / np.max(values)
    print("id = ", len(values[values > thresh]))
    return values


def MaxLikelihood(data, k=3):
    """
    Return estimation of dataset intrasic dimension using
    article: Elizaveta Levina, Peter J. Bickel
    Maximum Likelihood Estimation of Intrinsic Dimension"""
    # k is nbr of neirest neigbhors to take into account
    n = data.shape[1]
    Matrix = np.zeros((n, n))
    # Compute Euclidian distance
    for i in range(n):
        for j in range(n):
            Matrix[i, j] = np.linalg.norm(data[:, i] - data[:, j])
    Matrix.sort()  # sort by Nearest Neigh
    mk = 0  # computed dimension
    for x in range(n):
        mkx = 0
        for j in range(1, k):  # to k-1
            mkx += np.log(Matrix[x, k] / Matrix[x, j])
        mk += (1 / (k - 1) * mkx) ** (-1)
    mk = mk / n
    return mk


def FourierTransform(data, plot=True):
    """Return the fourier spectrum of the data"""
    if len(data.shape) < 2:
        data = np.expand_dims(data, 1)
    transforms = np.zeros(data.shape)
    for i in range(data.shape[1]):
        transforms[:, i] = np.abs(np.fft.fft(data[:, i]))
    transforms = transforms[: transforms.shape[0] // 2, :]
    if plot:
        plt.plot(transforms)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
    return transforms


### MUTUAL INFORMATION


def MutualMatrix(data, bins=50):
    """Computes mutual info matrix
    # todo : verfier code similaire à scipy
    """

    def MutualInfo(X, Y, bins):
        """Computes the mutal information bw two time series
        I(X, Y) = H(x) + H(Y) - H(X, Y) where H is shannon entropy"""
        # Compute histogram to get probabilities
        # numpy histo returns a tuple (hist, bins)
        hist_XY = np.histogram2d(X, Y, bins)[0]
        hist_X = np.histogram(X, bins)[0]
        hist_Y = np.histogram(Y, bins)[0]
        # Compute Entropy
        H_X = Entropy(hist_X)
        H_Y = Entropy(hist_Y)
        H_XY = Entropy(hist_XY)
        # Return Mutual Info
        return H_X + H_Y - H_XY

    def Entropy(hist):
        """Computes Shannon entropy from histogramm"""
        # Compute proba from histo
        p = hist / float(np.sum(hist))
        # Remove all zeros values
        # p*log(p) -> 0 when p->0
        p = p[p > 0]
        # Compute Entropy
        H = -sum(p * np.log2(p))
        return H

    # Matrix of all pairwise mutual info
    n = data.shape[1]  # nbr modes
    Matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Matrix[i, j] = MutualInfo(data[:, i], data[:, j], bins=bins)
        Matrix[i, i] = 2.6
    # plt.pcolormesh(M)
    return Matrix


# scp mergny@grappe:/workdir/mergny/OUTPUT/time_coefficients.d ~/Documents/Work/PLR2/LSTM_mergny/LSTM_build/Data/time_coefficient.d

#%%
### MAIN
if __name__ == "__main__":
    nbr_mods = 100
    start_mod = 1
    data = ImportData(file_name="Data/time_coefficient.d", modes=range(start_mod, start_mod + nbr_mods), nbr_snaps=-1)
