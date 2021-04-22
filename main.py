# %% Imports and Defining Dataset
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:28:56 2021
@author: Cyril
"""
### IMPORTS
#%reload_ext autoreload
#%autoreload 2

import torch
import numpy as np
import model as lstm
import matplotlib.pyplot as plt
import set_data as setdata
import plot_data as plotdata

### MAIN

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = 50
bs = 4
# Read and generate dataset
#data = setdata.ImportData(file_name='coeff', modes=range(m, m+20))
data = setdata.GenerateData(tf=4*np.pi, n=500, freq=range(4,15))
data_train, data_test = setdata.MakeDataset(data, split=0.7)
# Generate Inputs and Targets
iw, ow, stride = 120, 33, 10 # input window, output window, stride
x_train, y_train = setdata.WindowedDataset(data_train, iw, ow, stride, nbr_features=data_train.shape[1]) 
x_valid, y_valid = setdata.WindowedDataset(data_test, iw, ow, stride, nbr_features=data_test.shape[1])
#x_train = x_train + np.random.normal(0, 0.02, x_train.shape)
plt.plot(x_train[:, 0, 10])
# Convert tensor and set device
x_train, y_train, x_valid, y_valid = setdata.Convert2Torch(x_train, y_train, x_valid, y_valid, device=device)

# %% Defining and Training Model
model = lstm.LSTM_EncoderDecoder(input_size=x_train.shape[2], hidden_size=20).to(device)
#model = lstm.SimpleLSTM(input_size=x_train.shape[2], hidden_size=20).to(device)
loss = lstm.TrainModel(model, x_train, y_train, n_epochs=100, target_len=ow, batch_size=bs, learning_rate=0.03, wd=1e-9)
plt.plot(np.log10(loss))

# %% Valid Model
mode = 10
batch = 2
p_valid = lstm.Predict(model, x_valid[:, :bs, :], target_len=33)
plotdata.PlotPredictions(x_valid, y_valid, p_valid, batch, mode)

# %% Plot train
mode = 4
batch = 1
p_train = lstm.Predict(model, x_train[:, :bs, :], target_len=33)
plotdata.PlotPredictions(x_train, y_train, p_train, batch, mode)

# %% Saving and loading model
path = 'SavedModels/' + input('model name:')
torch.save(model.state_dict(), path)
#model.load_state_dict(torch.load(path))
#model.eval()

