# Main script for training the time series
# forecasting neural networks.
# Choose specific NN by uncommenting the model's line
#
# Author : Cyril Mergny
# Mail : cyril.mergny@ens-lyon.fr
# Github: https://github.com/Cyril-Mergny
#
# This project is part of a master degree internship
# Supervisors: Lionel Mathelin, Berangere Podvin
# Date of internship: Mars to September 2021 


### IMPORTS

# For saving 
import os
import glob
import shutil 

import trainer
import import_data
from models.LSTM_AE import LSTM_EncoderDecoder
from models.LSTM_A import LSTM_Attention
from models.transformers import Transformer
from models.multiscale import MultiScaleLSTMA

### FUNCTIONS

def create_savedir(overwrite=False):
    """
    Create a directory 'runs/run_0i/' to save 
    the trained model, infos and plots.  
    If ovewrite == False iterate dir name.
    """
    # List all runs dirs
    folders = glob.glob('runs/run_*')
    # Find next index for naming
    idx = int(max(folders).split('_')[1])+1 if len(folders)>0 else 1
    # Overwrite on folder run_01 if asked
    if overwrite and 'runs/run_01' in folders:
        shutil.rmtree('runs/run_01/')
        idx = 1
        print('Overwriting folder run_01...')
    # Create dir 
    saving_dir = 'runs/run_{:02d}/'.format(idx)
    os.makedirs(saving_dir)
    os.makedirs(saving_dir+'preds/')
    os.makedirs(saving_dir+'attention/')
    print(f'Created {saving_dir} directory.')
    return(saving_dir) # return dir path

### MAIN

# Create current run dir
saving_dir = create_savedir(overwrite=True) 

# Import and define dataset
data = import_data.Data(filename='data/spring_data.txt', modes=range(0, 10))
data.PrepareDataset(noise=True, in_out_stride=(200, 30, 100))


# Initialize model by uncommenting one line
H = 32 # Hidden size (H=4 for MA)
E = data.x_train.shape[2] # Input size
#model = LSTM_EncoderDecoder(E, H).to(data.device)
#model = Transformer(d_model=128, nhead=8).to(data.device)
#model = LSTM_Attention(E, H).to(data.device)
model = MultiScaleLSTMA(E, H).to(data.device)

# Load an retrain existing saved
#model.load_state_dict(torch.load(saving_dir+'best_model'))

# Training and saving model
trainer = trainer.Trainer(model, data)
trainer.train(epochs=80, bs=116, lr=1e-3, saving_dir=saving_dir)
