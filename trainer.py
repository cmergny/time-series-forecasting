import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange


class Trainer():
    """Trainer class for training."""
    
    def __init__(self, model, mydata) -> None:
        self.model = model
        self.data = mydata
        
    def step(self, input_tensor, target_tensor, n_batches, training=False):
        # PROBLEM if x.shape[1] is not divisible by bs then the element are not taken !!!
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
            #loss = self.compute_loss(outputs, target_batch)
            batch_loss += loss
            # Backpropagating 
            if training:
                    loss.backward()
                    self.optimizer.step()
        return(batch_loss/n_batches)
        
    def compute_loss(self, output, target):
        # (T, N, E) 
        loss = torch.sum(torch.abs((output-target)), dim=0)
        # (N, E)
        loss = torch.mean(loss)
        return(loss)
        
    def train(self, epochs, bs, lr, path):
        """Train the NN for a set number of epochs"""
        print('Training ...')
        self.path = path
        self.bs = bs
        self.epochs = epochs
        self.test_loss = np.full(epochs, np.nan) 
        self.valid_loss = np.full(epochs, np.nan) 
        self.best_loss = 1e10 # badly written I know
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        #self.optimizer = NoamOpt(d_model=1e4, warmup=300, optimizer=self.optimizer)
        self.criterion = nn.MSELoss()
        
        n_batches_train = int(self.data.x_train.shape[1]/bs) 
        n_batches_valid = int(self.data.x_valid.shape[1]/bs)
        
        # Normalisation necessary for NN training
        
        input_train = self.data.x_train.detach().clone().to(self.data.device)
        input_valid = self.data.x_valid.detach().clone().to(self.data.device)
        """
        for i in range(input_train.shape[2]):
            input_train[:, :, i] /= torch.max(torch.abs(input_train[:, :, i]))
            input_valid[:, :, i] /= torch.max(torch.abs(input_valid[:, :, i]))
        """
        
        with trange(epochs) as tr:
            for ep in tr:
                # training loop
                self.model.train()
                self.test_loss[ep] = self.step(input_train, self.data.y_train, n_batches_train, training=True) 
                # evaluation loop
                self.model.eval()
                with torch.no_grad():
                    self.valid_loss[ep] = self.step(input_valid, self.data.y_valid, n_batches_valid)
                # Every 10 ep check if valid loss is the best 
                if self.valid_loss[ep] < self.best_loss and ep%10==0:
                    # then save model
                    self.model.save(path+'best_model') 
                    self.best_loss = self.valid_loss[ep]               
                # Print on progress bar
                tr.set_postfix(train="{0:.2e}".format(self.test_loss[ep]),
                               valid="{0:.2e}".format(self.valid_loss[ep]))#,
                               #lr="{0:.2e}".format(self.optimizer._rate))
        # Export attention coeffs for exploitation
        self.save_run()
        return(self.test_loss, self.valid_loss)
        
    def __repr__(self) -> str:
        """Printing the Trainer class shows its important attributes"""
        text = '\nTraining : '
        text += '\n\tDevice = '+torch.cuda.get_device_name(self.data.device)
        text += f'\n\tNumber of epochs = {self.epochs}'
        text += f'\n\tBatch size = {self.bs}'
        text += f'\n\tBest valid loss = {self.best_loss:.3e}'
        
        text += f'\n\nModel : \n\t{self.model}'
        
        return(text)
    
    def plot_loss(self):
        """Plot and save the logarithm of the training and
            validation loss over all the epochs"""
        fig, ax = plt.subplots()
        ax.plot(np.log10(self.test_loss), label='Training set')
        ax.plot(np.log10(self.valid_loss), label='Validation set')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        plt.savefig(self.path+'loss')
        return None
    
    def save_run(self):
        """ Saves important infos and model
            in the summary.txt file"""
        with open(self.path+'summary.txt', 'w') as f:
            f.write(str(self.data))
            f.write(str(self))
        self.data.plot(self.path)
        self.plot_loss()
        return None
            

class NoamOpt:
    """Optimizer wrapper that implements adaptative rate
    as in Attention is all you need Paper"""
    def __init__(self, d_model, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = 1
        self.d_model = d_model
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return  self.d_model**(-0.5)*min(step**(-0.5), step*self.warmup**(-1.5))



### MAIN

if __name__ == '__main__':

    epochs = 500
    d_model = 1
    warmup = 20
    opt = NoamOpt(d_model, warmup, None)
    lr =  [opt.rate(i)  for i in range(1, epochs)]
    print('{0:.2e}'.format(max(lr)))
    
    # plot lr
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, epochs), lr)
    ax.set_yscale('log')
    plt.show()
    