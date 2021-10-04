# Time Series Forecasting with Neural Networks
--------

This repositery is the result of a master degree internship at the LISN lab on deep learning for fluid dynamics.
I've been interested at the different models that could be used for making time predictions on multivariate signals.
In particular, one interest of this study is to do forecasting on POD modes describing a turbulent flow. 

We started by building the classical LSTM Auto-Encoder from scratch, and then looked at more complex models involving the Attention mecanism.

 <p align="center">
<img src=misc/figures/Mode002Batch000.png width="400" />
</p>

## Requirements
--------

Python 3.x is required with *pytorch* installed. Having a cuda compatible GPU is strongly advised. You may also require to install *tqdm*, *glob* and *shutil* modules if not installed by default in your environnement.

The best, easiest and fastest way to do this, is by using conda and the environnement.yml file in this repo. In **the time-series-forecasting** folder open a terminal and type:

     conda env create -f misc/environment.yml -n TSF
     conda activate TSF
     
this will create a conda environnement called *TSF* with the necessary modules to run theses scripts. Once activated, run the scripts in the terminal.
*When creating the env pip might return an error, if so just ignore it and type:*

     pip install einops
     
*this seems to fix the issue.*
 

## Project layout
--------

Import your data, build the model, train and predict.

<ul>
<li> <b>import_data.py</b> : Data class used by main.py to import arrays from the data/ folder and organises the data into a pytorch compatible dataset. </li>

<li>  <b>main.py</b>  : calls the import data, defines and train a model. The model best weights are saved into a run/run_** folder.</li>

<li>  <b>trainer.py</b>  : class called by main.py to train and a save a model. Mean square error is used to compute the training and validation losses.</li>

<li>  <b>predict.py</b>  : requires a the saving file of a pretrained model. Allows to make prediction on 
unseen data. Plots the results in the model directory. </li>
</ul>

## How to use

### Training

To train a model, **first import data** from your own files or one of the exemples in the data/ folder:

     # Imports the first 10 variables of the data array
    data = import_data.Data(filename, modes=range(0, 10)) # shape (S, E)
   
The script uses a Data method to generate a pytorch compatible dataset. Each mode mode will be split into multiple sub windows to form a batch

    input_window = 100
    output_window = 30
    stride = 50
    data.prepare_dataset(in_out_stride=(input_window, output_window, stride))

Then, it **creates an instance of a model** with one of the uncommented model's lines. For exemple, to use LSTM with Attention model:

    model = LSTM_Attention(E, H).to(data.device)

The model is ready, the training process is called by the following lines:

    epochs = 200
    bs = 64 # error if bs is > dataset batch dim
    lr = 1e-3
    trainer = trainer.Trainer(model, data)
    trainer.train(epochs=epochs, bs=bs, lr=lr, saving_dir=saving_dir)

Once the main.py script is ready to run, type in a terminal:

    python main.py

and at the end of the training process, the trained model is saved in the current runs/run_** directory. Other useful infos are also stored in this folder, like the losses plot, and the data structured used.

### Predicting

The prediction occurs by running the predict.py script

    python predict.py

Specify the file on the run you want to use:
    path = 'runs/my_run/'

If using the same dataset but with the testing set, make sure to import data **EXACTLY** as done in the main.py file.
Also, when calling an instance make sure to use the **SAME MODEL WITH SAME HYPERPARAMETERS** as used for the training process.
In most cases, this is simply done by copy pasting the data and model calls lines from main.py to predict.py.

Model weights and testing data are loaded with the following lines:

    # Data to predict
    # Change it to any data of same shape/device
    x = data.x_valid # (S, :, E)
    y = data.y_valid # (T, :, E)
    # Load saved model
    predicter = Predicter(model, path, x)
    predicter.load_weights()

Then multiples predictions are made by the NN using the predicter method multi_pred
    T = 30 # nbr of timesteps to predict
    predicter.multi_pred(target_len=T)

Finally for model that support it, the predicter class can also plot attention weights:

    if model.name in ['LSTM_A', 'Multiscale']:
        predicter.plot_attention()

Predictions and Attention Weights plots are saved in the loaded model directory in the preds/ and attention/ sub folders.

## More Infos

For more infos and figures on the model structures and training please take a look at the misc/summary.pdf file. 
