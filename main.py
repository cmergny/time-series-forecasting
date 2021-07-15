### IMPORTS

import os
import glob 

import training
import utils.import_data as import_data

from models.LSTM_AE import LSTM_EncoderDecoder
from models.LSTM_A import LSTM_Attention
from models.transformers import RealTransfo
from models.multiscale import MultiScaleLSTMA

### FUNCTIONS

def makedir():
    """ Create a directory for saving"""
    liste = glob.glob('runs/run_*')
    idx = int(max(liste).split('_')[1])+1 if len(liste)>0 else 1
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
path = makedir() # used for saving
data = import_data.Data(filename='data/coeff', modes=range(10, 50), multivar=False)
data.PrepareDataset(in_out_stride=(80, 1, 5))
print(data)

# Create Model
#model = RealTransfo(d_model=128, nhead=8).to(mydata.device)
model = LSTM_Attention(data.x_train.shape[2], 32).to(data.device)
#model = MultiScaleLSTMA(data.x_train.shape[2], 16).to(data.device)
trainer = training.Trainer(model, data)

# Training
trainer.train(epochs=40, bs=8, lr=8e-4, path=path)

# Save
save_run(path, data, trainer)
