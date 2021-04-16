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

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = 0

# Read and generate Dataset
Data = setdata.ReadCoeff(file_name='coeff')
Data = setdata.TruncateNormData(Data, modes=range(m, m+10), nbr_snaps=300, index=2)
# Split into Training and Testing sets
Data_train, Data_test = setdata.MakeDataset(Data, split=0.7)

# Generate Inputs and Targets
iw, ow, stride = 110, 30, 10 # input window, output window, stride
x_train, y_train = setdata.WindowedDataset(Data_train, iw, ow, stride, nbr_features=Data_train.shape[1]) 
x_valid, y_valid = setdata.WindowedDataset(Data_test, iw, ow, stride, nbr_features=Data_train.shape[1])
x_train = x_train + np.random.normal(0, 0.08, x_train.shape)
# Convert tensor and set device
x_train, y_train, x_valid, y_valid = setdata.Convert2Torch(x_train, y_train, x_valid, y_valid, device=device)


# %% Defining and Training Model
model = lstm.LSTM_EncoderDecoder(input_size=x_train.shape[2], hidden_size=20).to(device)
# Train model
#loss = model.train_model(x_train, y_train, x_valid, y_valid, n_epochs=300, target_len=ow, batch_size=8, learning_rate=0.02, wd=1e-7, device=device)
#plt.plot(np.log10(loss))

# %% Test Model
# Predict 
batch, mode = 0, 9
title = f"Test set : mode {m+mode}"
name = f"test_mode{m+mode}.jpg"
Ptest = model.predict(input_tensor=x_valid[:, batch, :], target_len=ow)
Ptest = Ptest.reshape(Ptest.shape[0], 1, Ptest.shape[-1])
plotdata.PlotPredictions(x_valid, y_valid, Ptest, batch, mode=mode, title=title, name=name)

# %% Plot train
batch, mode = 2, 9
title = f"Training set : mode {m+mode}"
name = f"train_mode{m+mode}.jpg"
Ptrain = model.predict(input_tensor=x_train[:, batch, :], target_len=ow)
Ptrain = Ptrain.reshape(Ptrain.shape[0], 1, Ptrain.shape[-1])
plotdata.PlotPredictions(x_train, y_train, Ptrain, batch, mode=mode, title=title, name=name)


# %% Saving and loading model
path = 'SavedModels/' + input('model name:')
torch.save(model.state_dict(), path)

# %% Loading model
model_name = 'overfit_highmodes'
model.load_state_dict(torch.load(f'SavedModels/{model_name}'))
model.eval()