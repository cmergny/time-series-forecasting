#%% IMPORTS
import matplotlib.pyplot as plt
import utils.import_data as import_data
import models.LSTM_AE as LSTM_AE
import models.transformers as transfo
import utils.plot_data as plot_data
import training

%reload_ext autoreload
%autoreload 2
%matplotlib inline

# %% Prepare Data
mydata = import_data.Data()
mydata.PrepareDataset(in_out_stride=(100, 20, 50))
print(mydata)

#%% Define and train model
bs = 16
#model = transfo.RealTransfo(d_model=128, nhead=8).to(mydata.device)
model = LSTM_AE.LSTM_EncoderDecoder(mydata.x_train.shape[2], 32).to(mydata.device)

trainer = training.Trainer(model, mydata)
test_loss, valid_loss = trainer.train(epochs=200, bs=bs, lr=1e-3)
print(trainer)

# %% Predict
batch = 3
input_batch = mydata.x_valid[:, batch:batch+bs, :] 
target_batch = mydata.y_valid[:, batch:batch+bs, :] 
p_valid = training.Predict(model, input_batch=input_batch, target_len=20)
plot_data.PlotPredictions(input_batch, target_batch, p_valid, batch=batch, mode=0)

# %% Saving and loading model
#path = 'SavedModels/' + input('model name:')
#torch.save(model.state_dict(), path)
#model.load_state_dict(torch.load('SavedModels/overfit_butperfect'))
#model.eval()

