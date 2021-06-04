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
# Read and generate dataset
m = 1
ow = 20
data = setdata.ImportData(file_name='Data/coeff',  modes=range(m, m+150))
#data = setdata.GenerateData(L=300, nbr_samples=300)
#data =  setdata.AirQualityData(width=2000)
x_train, y_train, x_valid, y_valid = setdata.PrepareDataset(data, noise=None, in_out_stride=(80, ow, 30))
#plt.plot(x_train.to('cpu').detach()[:,0,0])

# %% Create and train model
bs = 16
model = lstm.LSTM_EncoderDecoder(input_size=x_train.shape[2], hidden_size=50).to(x_train.device)
loss = lstm.TrainModel(model, x_train, y_train, x_valid, y_valid, n_epochs=50, target_len=ow, batch_size=bs, lr=1e-2, wd=1e-5)
print(torch.cuda.get_device_name(x_train.device))
plt.plot(np.log10(loss))

# %% Predict&Plot on valid 
inlen = -0
batch = 110
input_batch = x_valid[inlen:, batch:batch+bs, :]
p_valid = lstm.Predict(model, input_batch, target_len=ow)
plotdata.PlotPredictions(input_batch, y_valid[inlen:, batch:batch+bs, :], p_valid, batch=0, mode=0)


#%% Predict&Plot on train data
p_train = lstm.Predict(model, x_train[inlen:, :bs, :], target_len=ow)
plotdata.PlotPredictions(x_train[inlen:], y_train, p_train, batch=4, mode=0)

# %% Saving and loading model
#path = 'SavedModels/' + input('model name:')
#torch.save(model.state_dict(), path)

model.load_state_dict(torch.load('SavedModels/overfit_butperfect'))
model.eval()

