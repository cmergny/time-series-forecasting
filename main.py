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
    """ Create a directory for saving"""
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

def save_run(path, data, trainer):
    """ Saves important infos and model"""
    with open(path+'summary.txt', 'w') as f:
        f.write(str(data))
        f.write(str(trainer))
    data.plot(path)
    trainer.plot_loss()

### MAIN

# Create Dataset
path = makedir(overwrite=True) # used for saving
data = import_data.Data(filename='data/coeff', modes=range(70, 120), multivar=True)
data.PrepareDataset(in_out_stride=(80, 10, 1))
print(data)

# Create Model
#model = LSTM_EncoderDecoder(data.x_train.shape[2], 32).to(data.device)
#model = RealTransfo(d_model=128, nhead=8).to(data.device)
#model = LSTM_Attention(data.x_train.shape[2], 32).to(data.device)
model = MultiScaleLSTMA(data.x_train.shape[2], 32).to(data.device)
trainer = trainer.Trainer(model, data)

# Training
trainer.train(epochs=500, bs=48, lr=8e-4, path=path)

# Save
save_run(path, data, trainer)
