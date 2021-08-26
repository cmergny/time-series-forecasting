### IMPORTS

import os
import glob
import shutil 

import trainer
import import_data

from models.LSTM_AE import LSTM_EncoderDecoder
from models.LSTM_A import LSTM_Attention
from models.transformers import RealTransfo
from models.multiscale import MultiScaleLSTMA

### FUNCTIONS

def makedir(overwrite=False):
    """ Create a directory called run_xx for saving"""
    # list all runs
    folders = glob.glob('runs/run_*')
    # find next indice for naming
    idx = int(max(folders).split('_')[1])+1 if len(folders)>0 else 1
    # overwrite on folder run_01 if asked
    if overwrite and 'runs/run_01' in folders:
        shutil.rmtree('runs/run_01/')
        idx = 1
        print('Overwriting folder run_01...')
    # Create dirs 
    path = 'runs/run_{:02d}/'.format(idx)
    os.makedirs(path)
    os.makedirs(path+'preds/')
    print(f'Created {path} directory.')
    return(path)


### MAIN

# Create current run dir
path = makedir(overwrite=True) 
# Import and define dataset
data = import_data.Data(filename='data/spring_data.txt', modes=range(0, 8), multivar=True)
data.PrepareDataset(in_out_stride=(100, 20, 50))

# Create Model
model = LSTM_EncoderDecoder(data.x_train.shape[2], 32).to(data.device)
#model = RealTransfo(d_model=128, nhead=8).to(data.device)
#model = LSTM_Attention(data.x_train.shape[2], 32).to(data.device)
#model = MultiScaleLSTMA(data.x_train.shape[2], 32).to(data.device)

# Training and saving model
trainer = trainer.Trainer(model, data)
trainer.train(epochs=100, bs=4, lr=8e-4, path=path)

