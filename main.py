#%% IMPORTS
import matplotlib.pyplot as plt
import utils.import_data as import_data
import models.LSTM_AE as LSTM_AE
import models.LSTM_A as LSTM_A
import models.transformers as transfo
import utils.plot_data as plot_data
import torch
import training

#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline

# Prepare Data
mydata = import_data.Data(file_name='data/coeff', modes=range(150, 180))
mydata.PrepareDataset(in_out_stride=(100, 20, 50))
print(mydata)

# Create Model
#model = transfo.RealTransfo(d_model=128, nhead=8).to(mydata.device)
model = LSTM_A.LSTM_Attention(mydata.x_train.shape[2], 32).to(mydata.device)
trainer = training.Trainer(model, mydata)
bs = 16


#%% Define and train model
epochs = 5000
test_loss, valid_loss = trainer.train(epochs=epochs, bs=bs, lr=8e-4)
print(trainer)
plt.savefig('cluster/loss.png')

# %% Predict
#model.load_state_dict(torch.load('saved_models/best_model'))
batch = 7
input_batch = mydata.x_valid[:, batch:batch+bs, :] 
target_batch = mydata.y_valid[:, batch:batch+bs, :] 
p_valid = trainer.predict(x=input_batch, target_len=20)
plot_data.PlotPredictions(input_batch, target_batch, p_valid, batch=3, mode=0)
#model.save('saved_models/check')
