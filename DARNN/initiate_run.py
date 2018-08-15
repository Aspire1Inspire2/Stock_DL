
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

import utility as util

global logger


import pickle
import numpy as np
from torch.utils.data import DataLoader

from DARNN import encoder, decoder
from data_class import StockDataset

import argparse

parser = argparse.ArgumentParser('Train the model using MNIST dataset.')
parser.add_argument('--data', default=
                    'data/python_stock_data.pickle',
                    required=False,
                    help='The path to save MNIST dataset, or '
                         'the path the dataset is located')
parser.add_argument('--model', default='lstm', required=False,
                    choices=['lstm', 'bnlstm'],
                    help='The name of a model to use')
parser.add_argument('--save', default='.', required=False,
                    help='The path to save model files')
parser.add_argument('--hidden-size', default=3, required=False,
                    type=int,
                    help='The number of hidden units')
parser.add_argument('--pmnist', default=False, action='store_true',
                    help='If set, it uses permutated-MNIST dataset')
parser.add_argument('--batch-size', default=700, required=False, type=int,
                    help='The size of each batch')
parser.add_argument('--max-iter', default=3, required=False, type=int,
                    help='The maximum iteration count')
parser.add_argument('--gpu', default=True, action='store_true',
                    help='The value specifying whether to use GPU')
args = parser.parse_args()



#Hyperparameter
T = 252 # 252 trading days per year
BATCH_SIZE = 2 # test batch size
y_label = 10001 # The stock PERMNO label, 
                #this is the stock to be studied
DATA_PATH = args.data
model_name = args.model
save_dir = args.save
HIDDEN_SIZE = args.hidden_size
pmnist = args.pmnist
BATCH_SIZE = args.batch_size
max_iter = args.max_iter
use_gpu = args.gpu
torch.manual_seed(1)


device = torch.device("cuda:0" if use_gpu else "cpu")
if use_gpu:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

#Load the original data
stock_data = pickle.load(open(DATA_PATH, 'rb'))

# Set the research begin and end date:
temp = stock_data.loc['2002-12-31':'2012-12-31'].reset_index()

# From the semi-one-dimension table create a two dim table
# This two dimension table is used in the Pytorch dataset as input
temp = temp.pivot(index='DATE', columns='PERMNO', values=['RET','VOL'])
temp = temp.swaplevel(axis=1).sort_index(axis=1)

# Here is how to use it:
# 1. Input the processed two dim table "temp"
# 2. Input the number of trading days in lstm input
# 3. Input the device name
stock_dataset = StockDataset(temp, T, y_label, device)

# Let's try out the first sample
# Compare to temp[10001].iloc[0:253] to see it successfully loaded
#x, y, target = stock_dataset.__getitem__(0)
#print(x.size())
#print(y.size())
#print(target.size())

# Assign the Dataloader to automatically load batched data for you
stock_dataloader = DataLoader(dataset=stock_dataset, batch_size=BATCH_SIZE,
                              shuffle=False)

# Lets try out the dataloader
data_iter = stock_dataloader.__iter__()
data_iter.__init__(stock_dataloader)
#x_batch, y_batch, target_batch = data_iter.__next__()







# List hyperparameters here
TRAIN_SIZE = 1
#TEST_SIZE = 2
INPUT_DIM = 2
#HIDDEN_DIM = 14
PREDICT_DIM = 1
#HISTORY_DIM = 180

#    if torch.cuda.is_available():
#        MINIBATCH_SIZE = int(BATCH_SIZE / torch.cuda.device_count())
#    else:
#        MINIBATCH_SIZE = BATCH_SIZE

stock_dataset = StockDataset(temp, T, y_label, 'cpu')

# Let's try out the first sample
# Compare to temp[10001].iloc[0:253] to see it successfully loaded
x, y, target = stock_dataset.__getitem__(0)
print(x.size())
print(y.size())
print(target.size())

# Assign the Dataloader to automatically load batched data for you
stock_dataloader = DataLoader(dataset=stock_dataset, batch_size=BATCH_SIZE,
                              shuffle=False)

# Lets try out the dataloader
data_iter = stock_dataloader.__iter__()
data_iter.__init__(stock_dataloader)
x_batch, y_batch, target_batch = data_iter.__next__()


encoder = encoder(input_size = X.shape[1], 
                  hidden_size = encoder_hidden_size, 
                  T = T,
                  logger = logger)
decoder = decoder(encoder_hidden_size = encoder_hidden_size,
                  decoder_hidden_size = decoder_hidden_size,
                  T = T, 
                  logger = logger)

if parallel:
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()),
                                   lr = learning_rate)
decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, decoder.parameters()),
                                   lr = learning_rate)
# learning_rate = learning_rate

train_size = int(X.shape[0] * 0.7)
y = y - np.mean(y[:train_size]) # Question: why Adam requires data to be normalized?
logger.info("Training size: %d.", train_size)


def train(self, n_epochs = 10):
    iter_per_epoch = int(np.ceil(train_size * 1. / batch_size))
    logger.info("Iterations per epoch: %3.3f ~ %d.", train_size * 1. / batch_size, iter_per_epoch)
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)

    loss_func = nn.MSELoss()

    n_iter = 0

    #learning_rate = 1.

    for i in range(n_epochs):
        perm_idx = np.random.permutation(train_size - T)
        j = 0
        while j < train_size:
            batch_idx = perm_idx[j:(j + batch_size)]
            X = np.zeros((len(batch_idx), T - 1, X.shape[1]))
            y_history = np.zeros((len(batch_idx), T - 1))
            y_target = y[batch_idx + T]

            for k in range(len(batch_idx)):
                X[k, :, :] = X[batch_idx[k] : (batch_idx[k] + T - 1), :]
                y_history[k, :] = y[batch_idx[k] : (batch_idx[k] + T - 1)]

            loss = train_iteration(X, y_history, y_target)
            iter_losses[i * iter_per_epoch + int(j / batch_size)] = loss
            #if (j / batch_size) % 50 == 0:
            #    logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / batch_size, loss)
            j += batch_size
            n_iter += 1

            if n_iter % 10000 == 0 and n_iter > 0:
                for param_group in encoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
            '''
            if learning_rate > learning_rate:
                for param_group in encoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * .9
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * .9
                learning_rate *= .9
            '''


        epoch_losses[i] = np.mean(iter_losses[range(i * iter_per_epoch, (i + 1) * iter_per_epoch)])
        if i % 10 == 0:
            logger.info("Epoch %d, loss: %3.3f.", i, epoch_losses[i])

        if i % 10 == 0:
            y_train_pred = predict(on_train = True)
            y_test_pred = predict(on_train = False)
            #y_pred = np.concatenate((y_train_pred, y_test_pred))
            plt.figure()
            plt.plot(range(1, 1 + len(y)), y, label = "True")
            plt.plot(range(T , len(y_train_pred) + T), y_train_pred, label = 'Predicted - Train')
            plt.plot(range(T + len(y_train_pred) , len(y) + 1), y_test_pred, label = 'Predicted - Test')
            plt.legend(loc = 'upper left')
            plt.show()

def train_iteration(self, X, y_history, y_target):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_weighted, input_encoded = encoder(torch.tensor(X, dtype=torch.float, device=None, requires_grad=False))
    y_pred = decoder(input_encoded, torch.tensor(y_history, dtype=torch.float, device=None, requires_grad=False)).squeeze()

    y_true = torch.tensor(y_target, dtype=torch.float, device=None, requires_grad=False)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    # if loss.item() < 10:
    #     logger.info("MSE: %s, loss: %s.", loss.item(), (y_pred[:, 0] - y_true).pow(2).mean())

    return loss.item()

def predict(self, on_train = False):
    if on_train:
        y_pred = np.zeros(train_size - T + 1)
    else:
        y_pred = np.zeros(X.shape[0] - train_size)

    i = 0
    while i < len(y_pred):
        batch_idx = np.array(range(len(y_pred)))[i : (i + batch_size)]
        X = np.zeros((len(batch_idx), T - 1, X.shape[1]))
        y_history = np.zeros((len(batch_idx), T - 1))
        for j in range(len(batch_idx)):
            if on_train:
                X[j, :, :] = X[range(batch_idx[j], batch_idx[j] + T - 1), :]
                y_history[j, :] = y[range(batch_idx[j],  batch_idx[j]+ T - 1)]
            else:
                X[j, :, :] = X[range(batch_idx[j] + train_size - T, batch_idx[j] + train_size - 1), :]
                y_history[j, :] = y[range(batch_idx[j] + train_size - T,  batch_idx[j]+ train_size - 1)]

        y_history = torch.tensor(y_history, dtype=torch.float, device=None, requires_grad=False)
        _, input_encoded = encoder(torch.tensor(X, dtype=torch.float, device=None, requires_grad=False))
        y_pred[i:(i + batch_size)] = decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
        i += batch_size
    return y_pred