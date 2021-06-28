
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange

def TrainModel(model, input_tensor, target_tensor, x_valid, y_valid, n_epochs, 
               batch_size, lr, wd):

    # Loss and optimizer
    #writer = SummaryWriter()
    ow = 20
    Losses = np.full(n_epochs, np.nan) # Init losses array with NaNs
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    nbr_batches = int(input_tensor.shape[1] / batch_size) # nbr of batch iterations
    # Iterate by epochs
    with trange(n_epochs) as tr:
        for ep in tr:
            # training loop
            model.train()
            batch_loss = 0
            # Iterate by batch
            for b in range(nbr_batches):
                # Select batches
                input_batch = input_tensor[:, b:b+batch_size, :]
                target_batch = target_tensor[:, b:b+batch_size, :]
                # Initialise gradient to zero
                optimizer.zero_grad()
                # Calling model
                #outputs = model(input_batch[:-1], input_batch[-2:])
                outputs = model(input_batch, target_len=ow)
                # Computing loss
                loss = criterion(outputs, target_batch)
                batch_loss += loss
                # Backpropagating 
                loss.backward()
                optimizer.step()
            # Computing Loss FOR epoch
            batch_loss /= nbr_batches 
            #writer.add_scalar("Train_Loss", batch_loss, ep)
            Losses[ep] = batch_loss

            # evaluation loop
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                nbr_batches_valid = int(x_valid.shape[1]/batch_size) 
                for b in range(nbr_batches_valid):
                    input_batch = x_valid[:, b:b+batch_size, :]
                    target_batch = y_valid[:, b:b+batch_size :]
                    #outputs = model(input_batch[:-1], input_batch[-2:])
                    outputs = model(input_batch, target_len=ow)
                    loss = criterion(outputs, target_batch)
                    valid_loss += loss
                valid_loss = valid_loss/nbr_batches_valid
                #writer.add_scalar("Valid_Loss", valid_loss, ep)
            model.train()

            tr.set_postfix(train="{0:.2e}".format(batch_loss), valid="{0:.2e}".format(valid_loss)) 
    #writer.flush()        
    return(Losses)

    
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