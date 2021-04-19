# %% Imports and Defining Dataset
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:28:56 2021
@author: Cyril
"""
### IMPORTS
%reload_ext autoreload
%autoreload 2

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
bs = 16
# Read and generate dataset
#data = setdata.ImportData(file_name='coeff', modes=range(m, m+20))
data = setdata.GenerateData(tf=4*np.pi, n=500, freq=range(4,15))
data_train, data_test = setdata.MakeDataset(data, split=0.7)
# Generate Inputs and Targets
iw, ow, stride = 120, 100, 5 # input window, output window, stride
x_train, y_train = setdata.WindowedDataset(data_train, iw, ow, stride, nbr_features=data_train.shape[1]) 
x_valid, y_valid = setdata.WindowedDataset(data_test, iw, ow, stride, nbr_features=data_test.shape[1])
#x_train = x_train + np.random.normal(0, 0.02, x_train.shape)
plt.plot(x_train[:, 0, 10])
# Convert tensor and set device
x_train, y_train, x_valid, y_valid = setdata.Convert2Torch(x_train, y_train, x_valid, y_valid, device=device)

# %% Defining and Training Model
model = lstm.LSTM_EncoderDecoder(input_size=x_train.shape[2], hidden_size=20).to(device)
loss = model.train_model(x_train, y_train, x_valid, y_valid, n_epochs=200, target_len=ow, batch_size=bs, learning_rate=0.03, wd=1e-9, device=device)
plt.plot(np.log10(loss))

# %% Valid Model
# Predict 
batch, mode = 0, 4
inlen = 0
p_valid = model.predict(input_tensor=x_valid[-inlen:, batch, :], target_len=ow)
plotdata.PlotPredictions(x_valid[-inlen:,:,:], y_valid, p_valid, batch, mode)

# %% Plot train
batch, mode = 1, 9
ptrain = model.predict(input_tensor=x_train[:, batch, :], target_len=ow)
plotdata.PlotPredictions(x_train, y_train, ptrain, batch, mode)

# %% Test Model
data_test = setdata.GenerateData(tf=4*np.pi, n=500, freq=range(8,19))
data_test = data_test.reshape(data_test.shape[0], 1, data_test.shape[1])
x_test = data_test[:100, :, :]
y_test = data_test[100:100+ow, :, :]

x_test, y_test =  setdata.Convert2Torch(x_test, y_test, device=device)

batch, mode = 0, 1
inlen = 10
p_test = model.predict(input_tensor=x_test[-inlen:, batch, :], target_len=ow)
plotdata.PlotPredictions(x_test[-inlen:,:,:], y_test, p_test, batch, mode)


# %% Saving and loading model
path = 'SavedModels/' + input('model name:')
torch.save(model.state_dict(), path)
#model.load_state_dict(torch.load(path))
#model.eval()

