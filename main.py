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
m = 10
ow = 20
data = setdata.ImportData(file_name='Data/coeff',  modes=range(m, m+10))
#data = setdata.GenerateData(L=300, nbr_samples=300)
# overlap ?
#data = setdata.AirQualityData(width=2000)
x_train, y_train, x_valid, y_valid = setdata.PrepareDataset(data, noise=None, in_out_stride=(100, ow, 30))
plt.plot(x_train.to('cpu').detach()[:,0,0])
# demander ç berangere !
# rediger un peu l'explication sur le reshape


# %% Create and train model
bs = 32
model = lstm.LSTM_EncoderDecoder(input_size=x_train.shape[2], hidden_size=50).to(x_train.device)
#loss = lstm.TrainModel(model, x_train, y_train, x_valid, y_valid, n_epochs=300, target_len=ow, batch_size=bs, lr=1e-3, wd=1e-5)
plt.plot(np.log10(loss))

# %% Predict&Plot on valid and train data
inlen = -0
batch = 20
p_valid = lstm.Predict(model, x_valid[inlen:, batch:batch+bs, :], target_len=ow)
plotdata.PlotPredictions(x_valid[inlen:, batch:batch+bs, :], y_valid[inlen:, batch:batch+bs, :], p_valid, batch=0, mode=0, name='ylabels')

p_train = lstm.Predict(model, x_train[inlen:, :bs, :], target_len=ow)
plotdata.PlotPredictions(x_train[inlen:], y_train, p_train, batch=4, mode=0)

# %% Saving and loading model
#path = 'SavedModels/' + input('model name:')
#torch.save(model.state_dict(), path)

model.load_state_dict(torch.load('SavedModels/ok_iw100ow20'))
model.eval()

