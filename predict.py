# Used for predictions of trained models.
# Precise the saving dir path
# Define data and model EXACTLY like in main.py !
# Plots are saved in the saving directory.


### IMPORTS

import torch
import import_data
import numpy as np
import matplotlib.pyplot as plt

from models.LSTM_AE import LSTM_EncoderDecoder
from models.LSTM_A import LSTM_Attention
from models.transformers import Transformer
from models.multiscale import MultiScaleLSTMA


### Predicter class

class Predicter:
    """
    Predicter class for inference.
    Takes a model and input array to predict the next time steps.
    Plot predictions in the runs/preds dir.
    For compatabile models also plots attention in runs/attention dir.
    """
    
    def __init__(self, model, path, x, y=None) -> None:
        self.model = model
        self.path = path
        self.pred_path = path + 'preds/'
        self.att_path = path + 'attention/'
        # Input to predict
        self.x = x
        # (Optional) target reference
        self.y = y if y != None else torch.zeros((1, x.shape[1], x.shape[2]))
        
    def predict(self, x, target_len):
        """Call model in eval mode"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x, target_len)
            return(outputs)
        
    def multi_pred(self, target_len, autoregressive=False):
        """ Two methods for predicting multiple time steps in time:
        Default: Model is already trained to predict multiple steps.
        Autoreg: Model can only do one pred at a time so it iteratively
            feeds ouput as a new input.
        Also calls the plot prediction method.
        """
        x = self.x.clone().detach()
        if not autoregressive:
            # Predict directly T times
            p = self.predict(x, target_len)
        else:
            # Predict one time ahead and repeat 
            p = torch.zeros(target_len, x.shape[1], x.shape[2]).to(x.device) # (P, N, E)
            for i in range(target_len):
                p[i] = self.predict(x, 1)
                x = torch.cat((x[1:], p[i])) # shift input one step
        # Plot predictions
        self.plot_pred(p)
        return(p)
    
    def save_fig(self, figname):
        """Saves figure to figname path and print it."""
        plt.savefig(figname)
        print(f'Saved attention to {figname}')
 
    def plot_pred(self, pred, batch=0):
        """ Plot the predictions array P next to input X and target Y arrays
        and save it to current run directory.
        """
        self.x, self.y, pred = [np.array(a.to('cpu').detach()) for a in [self.x, self.y, pred]]
        # Plot for all modes
        for mode in range(self.x.shape[2]):
            figname = f'Mode{mode:03d}Batch{batch:03d}'
            # Convert to plotable arrays and select mode
            x, y, p = [a[:, batch, mode] for a in [self.x, self.y, pred]]
            #  Parameters
            S = x.shape[0] # source size
            T = y.shape[0] # target size
            P = p.shape[0]
            # Plot with crosses and lines
            figure, ax = plt.subplots()
            ax.plot(range(S), x, label='Input')
            ax.plot(range(S, S+T), y, color='green',label='Target')
            ax.plot(range(S, S+T), y,'x', color='green')
            ax.plot(range(S, S+P), p, color='orange', label='Prediction')
            ax.plot(range(S, S+P), p, 'x', color='orange')
            ax.set_ylabel('Amplitude')
            ax.set_xlabel('Timesteps')
            plt.title(f'Predictions of {p.size} time steps for {figname[4:7]}')
            plt.legend()
            # Save figure with name set by mode
            figname = self.pred_path+figname
            self.save_fig(figname)
    
    def plot_attention(self, batch=0):
        """ Plot the attention weights of the LSTM_A and Multiscale
        Neural netwotks. Figures are differ according to the model.
        Figures saved in the runs/attention/directory.
        """
        alphas = self.model.alphas.to('cpu').detach() # (S, N, T) or (E, N, E')
        # 1D Attention
        for context in range(alphas.shape[2]):
            fig1, ax1 = plt.subplots()
            ax1.plot(alphas[:, batch, context])
            # Temporal Attention
            if self.model.name == 'LSTM_A':
                ax1.set_xlabel('Timesteps involved for prediction')
                ax1.set_ylabel('Weights amplitude')
                plt.title(f'Attention weights for predicting timestep {context} of all modes')
                figname = self.att_path+f'Timestep{context:03d}Batch{batch}'
            # Scale Attention
            elif self.model.name == 'Multiscale':
                ax1.set_xlabel('Modes involved for prediction')
                ax1.set_ylabel('Weights amplitude')
                plt.title(f'Attention weights for predicting mode {context}')
                figname = self.att_path+f'Mode{context:03d}Batch{batch}'
            # Save
            self.save_fig(figname)
        # 2D attention
        fig2, ax2 = plt.subplots()
        ax2.pcolormesh(alphas[:, batch, :].transpose(0, 1))
        if self.model.name == 'LSTM_A':
            ax2.set_xlabel('Timesteps involved for prediction')
            ax2.set_ylabel('Timestep to predict')
            #plt.title(f'Attention weights for predicting batch {batch}')
        if self.model.name == 'Multiscale':
            ax2.set_xlabel('Modes involved for prediction')
            ax2.set_ylabel('Modes to predict')
        # Save
        self.save_fig(self.att_path+f'2D_attention')
        return(alphas)
    
    def load_weights(self):
        model.load_state_dict(torch.load(self.path+'best_model'))
        print(f'Loaded best_model from {self.path}')
        
### MAIN

if __name__ == '__main__':
        
    path = 'runs/run_01/'
    # Import and define dataset
    data = import_data.Data(filename='data/spring_data.txt', modes=range(0, 2))
    data.PrepareDataset(noise=True, in_out_stride=(200, 30, 100))

    # Initialize model by uncommenting one line
    H = 32 # Hidden size (H=4 for MA)
    E = data.x_train.shape[2] # Input size
    #model = LSTM_EncoderDecoder(E, H).to(data.device)
    #model = LSTM_Attention(E, H).to(data.device)
    #model = Transformer(d_model=128, nhead=8).to(data.device)
    model = MultiScaleLSTMA(E, H).to(data.device)
    
    
    # Data to predict
    # Change it to any data of same shape/device
    x = data.x_valid # (S, :, E)
    y = data.y_valid # (T, :, E)
    # Load saved model
    predicter = Predicter(model, path, x)
    predicter.load_weights()
    
    # Predictions
    predicter.multi_pred(target_len=data.ow)
    # Attention
    if model.name in ['LSTM_A', 'Multiscale']:
        predicter.plot_attention()

    