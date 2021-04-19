# %% Imports and Defining Dataset
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:28:56 2021
@author: Cyril
"""
### IMPORTS

import torch
import numpy as np
import model as lstm
import matplotlib.pyplot as plt
import set_data as setdata
import plot_data as plotdata

### MAIN
%reload_ext autoreload
%autoreload 2
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = 50
bs = 16
# Read and generate dataset
#data = setdata.Importdata(file_name='coeff', modes=range(m, m+10))
data = setdata.GenerateData(tf=50*np.pi, n=2000, freq=range(1,6))
data_train, data_test = setdata.MakeDataset(data, split=0.7)
# Generate Inputs and Targets
iw, ow, stride = 500, 80, 50 # input window, output window, stride
x_train, y_train = setdata.WindowedDataset(data_train, iw, ow, stride, nbr_features=data_train.shape[1]) 
x_valid, y_valid = setdata.WindowedDataset(data_test, iw, ow, stride, nbr_features=data_test.shape[1])
x_train = x_train + np.random.normal(0, 0.05, x_train.shape)
plt.plot(x_train[:,1,:])
# Convert tensor and set device
x_train, y_train, x_valid, y_valid = setdata.Convert2Torch(x_train, y_train, x_valid, y_valid, device=device)

# %% Defining and Training Model
model = lstm.LSTM_EncoderDecoder(input_size=x_train.shape[2], hidden_size=10).to(device)
loss = model.train_model(x_train, y_train, x_valid, y_valid, n_epochs=500, target_len=ow, batch_size=8, learning_rate=0.02, wd=1e-7, device=device)
plt.plot(np.log10(loss))

# %% Test Model
# Predict 
batch, mode = 7, 4
ptest = model.predict(input_tensor=x_valid[:, batch, :], target_len=ow)
plotdata.PlotPredictions(x_valid, y_valid, ptest, batch, mode)

# %% Plot train
batch, mode = 3, 1
ptrain = model.predict(input_tensor=x_train[:, batch, :], target_len=ow)
plotdata.PlotPredictions(x_train, y_train, ptrain, batch, mode)

# %% Saving and loading model
#path = 'SavedModels/' + input('model name:')
# torch.save(model.state_dict(), path)
#model.load_state_dict(torch.load(path))
#model.eval()