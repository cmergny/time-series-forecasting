import matplotlib.pyplot as plt
import numpy as np
import torch
import import_data

from models.LSTM_AE import LSTM_EncoderDecoder
from models.LSTM_A import LSTM_Attention
from models.transformers import RealTransfo
from models.multiscale import MultiScaleLSTMA


### Predicter class
class Predicter:
    
    def __init__(self, model, data, path) -> None:
        self.model = model
        self.data = data
        self.path = path + 'preds/'
        
    def predict(self, **kwargs):
        """Call model in eval mode"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**kwargs)
            return(outputs)
        
    def autoreg_pred(self, batch, mode, target_len):
        """ Predit for multiple values"""
        print('Predicting...')
        # Select input arrays
        x = self.data.x_valid[:, batch, :].unsqueeze(1) # (S, 1, E)
        y = self.data.y_valid[:, batch, :].unsqueeze(1) # (S, 1, E)
        p = torch.zeros(target_len, x.shape[1], x.shape[2]).to(x.device)
        # Predict one time ahead and repeat 
        for i in range(target_len):
            p_valid = self.predict(x=x, target_len=1)
            p[i] = p_valid
            x = torch.cat((x[1:], p_valid))
        # Find target array using quick maths
        idx = target_len // self.data.stride + 1
        end =  -(self.data.stride- target_len%self.data.stride)
        y = self.data.x_valid[end-target_len:end, batch+idx, :].unsqueeze(1) 
        # Plot 
        x = self.data.x_valid[:, batch, :].unsqueeze(1) # (S, 1, E)
        self.plot_predictions(x, y, p, mode=mode, batch=batch)        
    
    def mutltistep_pred(self, batch, bs, target_len):
        x = self.data.x_valid[:, batch:batch+bs, :] # (S, N, E)
        y = self.data.y_valid[:, batch:batch+bs, :] # (S, N, E)
        # Predict and plot
        p = self.predict(x=x, target_len=target_len)
        self.plot_predictions(x, y, p, mode=0, batch=batch)    
        
    
    def plot_predictions(self, X, Y, P, mode, batch):
        """ Plot a prediction with the input and target curves"""
        # Convert to plotable arrays
        convert = lambda x: np.array(x.to('cpu').detach())
        if type(X) == torch.Tensor:
            X, Y, P = convert(X), convert(Y), convert(P)
        # retrieve important params
        len_x = len(X[:, mode])
        ow = Y.shape[0]
        target_len = P.shape[0]
        X, Y, P = X[:, mode], Y[:, mode], P[:, mode]
        # Plot with dots and curves
        figure, ax = plt.subplots()
        ax.plot(range(len_x), X, label='Input')
        ax.plot(range(len_x, len_x+ow), Y, color='green')
        ax.plot(range(len_x, len_x+ow), Y,'x', color='green',label='target')
        ax.plot(range(len_x, len_x+target_len), P, color='orange')
        ax.plot(range(len_x, len_x+target_len), P, 'x', color='orange', label='predictions')
        ax.set_ylabel('amplitude')
        ax.set_xlabel('timesteps')
        plt.title(f'Predictions of {target_len} time steps for batch {batch} and mode {mode}.')
        plt.legend()
        figname = self.path+f'pred_b{batch:03d}_m{mode:03d}'
        figure.savefig(figname)
        print(f'Saved prediction to {figname}')


    def plot_attention(self, model, batch=0, mode=0):
        """ Plot attention weights"""
        # 2D attention
        alphas = model.alphas.to('cpu').detach()
        fig, ax = plt.subplots()
        ax.pcolormesh(alphas[:, batch, :].transpose(0, 1))
        ax.set_xlabel('Modes involved for prediction')
        ax.set_ylabel('Modes to predict')
        plt.title(f'Attention weights for predicting batch {batch} mode {mode}')
        plt.savefig(self.path+f'2D_attention')
        # one mode attention
        fig, ax = plt.subplots()
        ax.plot(alphas[:, batch, mode])
        ax.set_xlabel('Modes involved for prediction')
        ax.set_ylabel('weights amplitude')
        plt.savefig(self.path+f'1D_attention')
        return(alphas)
    
    def plot_single_attention(self, model, batch=0, timestep=0):
        """ Plot attention weights"""
        # 2D attention
        alphas = model.alphas.to('cpu').detach() # (S, N, T)
        fig, ax = plt.subplots()
        ax.pcolormesh(alphas[:, batch, :].transpose(0, 1))
        ax.set_xlabel('Timesteps involved for prediction')
        ax.set_ylabel('Timestep to predict')
        plt.title(f'Attention weights for predicting batch {batch}')
        plt.savefig(self.path+f'2D_attention')
        # one mode attention
        fig, ax = plt.subplots()
        ax.plot(alphas[:, batch, timestep])
        ax.set_xlabel('Timesteps involved for prediction')
        ax.set_ylabel('weights amplitude')
        plt.title(f'Attention weights for predicting timestep {timestep} of batch {batch}')
        plt.savefig(self.path+f'1D_attention')
        return(alphas)

### MAIN

if __name__ == '__main__':
        
    path = 'runs/run_01/'
    # Create Dataset
    data = import_data.Data(filename='data/coeff', modes=range(100, 120), multivar=False)
    data.PrepareDataset(in_out_stride=(80, 10, 40))

    # Create Model
    #model = LSTM_EncoderDecoder(data.x_train.shape[2], 32).to(data.device)
    #model = RealTransfo(d_model=128, nhead=8).to(data.device)
    model = LSTM_Attention(data.x_train.shape[2], 32).to(data.device)
    #model = MultiScaleLSTMA(mydata.x_train.shape[2], 16).to(mydata.device)
    
    # Load
    predicter = Predicter(model, data, path)
    model.load_state_dict(torch.load(path+'best_model'))
    print(f'Loaded best_model from {path}')

    # Predict
    #predicter.autoreg_pred(batch=11, mode=0, target_len=10)
    predicter.mutltistep_pred(batch=35, bs=32, target_len=data.ow)
    alphas = predicter.plot_single_attention(model, batch=0)

