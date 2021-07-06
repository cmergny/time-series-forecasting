import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange


class Trainer():
    
    def __init__(self, model, mydata) -> None:
        self.model = model
        self.data = mydata
        
    def step(self, input_tensor, target_tensor, n_batches, training=False):
        """ Per batch training step"""
        batch_loss = 0
        # Iterate by batch
        for b in range(n_batches):
            # Init grad during train
            if training:
                self.optimizer.zero_grad()
            # Select batch
            input_batch = input_tensor[:, b:b+self.bs, :]
            target_batch = target_tensor[:, b:b+self.bs, :]
            # Calling model
            outputs = self.model(input_batch, target_len=self.data.ow)
            loss = self.criterion(outputs, target_batch)
            batch_loss += loss
            # Backpropagating 
            if training: 
                    loss.backward()
                    self.optimizer.step()
        return(batch_loss/n_batches)
        
    def train(self, epochs, bs, lr):
        
        self.bs = bs
        self.test_loss = np.full(epochs, np.nan) 
        self.valid_loss = np.full(epochs, np.nan) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-7)
        self.criterion = nn.MSELoss()
        
        n_batches_test = int(self.data.x_train.shape[1]/bs) 
        n_batches_valid = int(self.data.x_valid.shape[1]/bs) 
        
        with trange(epochs) as tr:
            for ep in tr:
                # training loop
                self.model.train()
                self.test_loss[ep] = self.step(self.data.x_train, self.data.y_train, n_batches_test, training=True) 
                    
                # evaluation loop
                self.model.eval()
                with torch.no_grad():
                    self.valid_loss[ep] = self.step(self.data.x_valid, self.data.y_valid, n_batches_valid)
                                    
                # Print on progress bar
                tr.set_postfix(train="{0:.2e}".format(self.test_loss[ep]), valid="{0:.2e}".format(self.valid_loss[ep])) 
        return(self.test_loss, self.valid_loss)
        
        
    def __repr__(self) -> str:
        fig, ax = plt.subplots()
        ax.plot(np.log10(self.test_loss), label='Training set')
        ax.plot(np.log10(self.valid_loss), label='Validation set')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        plt.plot()
        return('\n device = '+torch.cuda.get_device_name(self.data.device))
        
    
    
    
def Inference(model, input_batch, ow=20):
    """ Predicitons for a transformer """
    model.eval() 
    predictions = torch.zeros(ow, input_batch.shape[1], input_batch.shape[2]).to(input_batch.device)
    outputs = input_batch[0, :, :].unsqueeze(0)
    predictions[0] = outputs
    # Autoregressive
    for i in range(ow):
        with torch.no_grad():
            pred_batch = model(input_batch, outputs)
        # Append last prediction
        predictions[1:i+1, :, :] = pred_batch
        outputs = predictions[:i+1, :, :]
    # return all but first element
    return(predictions.detach().to('cpu')[1:])
    
    
def Predict(model, **kwargs):
    model.eval()
    with torch.no_grad():
        outputs = model(**kwargs)
        return(outputs.cpu().detach())