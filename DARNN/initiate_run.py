
import torch
from torch import nn
from torch import optim
#import torch.nn.functional as F
#
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

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
parser.add_argument('--hidden-size', default=64, required=False,
                    type=int,
                    help='The number of hidden units')
parser.add_argument('--pmnist', default=False, action='store_true',
                    help='If set, it uses permutated-MNIST dataset')
parser.add_argument('--batch-size', default=2, required=False, type=int,
                    help='The size of each batch')
parser.add_argument('--n_epochs', default=3, required=False, type=int,
                    help='The maximum iteration count')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='The value specifying whether to use GPU')
args = parser.parse_args()



#Hyperparameter
T = 252 # 252 trading days per year, 126 per half year
y_label = 88331 # The stock PERMNO label, 
                #this is the stock to be studied
DATA_PATH = args.data
model_name = args.model
save_dir = args.save
HIDDEN_SIZE = args.hidden_size
pmnist = args.pmnist
BATCH_SIZE = args.batch_size
n_epochs = args.n_epochs
use_gpu = args.gpu
torch.manual_seed(1)

learning_rate = 0.1
TRAIN_SIZE = 1
#TEST_SIZE = 2
INPUT_DIM = 2
#HIDDEN_DIM = 14
PREDICT_DIM = 1
#HISTORY_DIM = 180

device = torch.device("cuda:0" if use_gpu else "cpu")
if use_gpu:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

# #Load the original data
# stock_data = pickle.load(open(DATA_PATH, 'rb'))
#
# # Set the research begin and end date:
# temp = stock_data.loc['2002-12-31':'2012-12-31'].reset_index()
#
# # From the semi-one-dimension table create a two dim table
# # This two dimension table is used in the Pytorch dataset as input
# temp = temp.pivot(index='DATE', columns='PERMNO', values=['RET','VOL'])
# temp = temp.swaplevel(axis=1).sort_index(axis=1)
#
# with open('data/input.pickle', 'wb') as handle:
#    pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the processed data to accerlate debugging
# Before using this line, use the above line to dump the data.
temp = pickle.load(open('data/input.pickle', 'rb'))

## Use the following block to select only part of the whole input data
#columns = np.unique(temp.columns.get_level_values(0).values)
#trading_value = pickle.load(open('data/trading_value.pickle', 'rb'))
#trading_value = trading_value.loc[columns] \
#                       .sort_values('AMNT', ascending = False)
#ranking = trading_value.index \
#                       .get_level_values(0) \
#                       .values
#cutoff = (trading_value.values.squeeze() / trading_value.iloc[0].values) > 1e-3
#keep = ranking[cutoff]
#temp = temp[keep]


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
x_batch, y_batch, target_batch = data_iter.__next__()

n_stock = int(x_batch.size(2)/2)

print("encoder input size: " + str(x_batch.size(2)))
print("hidden size: " + str(HIDDEN_SIZE))


encoder = encoder(n_stock = n_stock, 
                  batch_size = BATCH_SIZE,
                  hidden_size = HIDDEN_SIZE, 
                  T = T,
                  device = device)
decoder = decoder(encoder_hidden_size = HIDDEN_SIZE,
                  decoder_hidden_size = HIDDEN_SIZE,
                  T = T)

#if torch.cuda.is_available():
#    MINIBATCH_SIZE = int(BATCH_SIZE / torch.cuda.device_count())
#else:
#    MINIBATCH_SIZE = BATCH_SIZE
#if parallel:
#    encoder = nn.DataParallel(encoder)
#    decoder = nn.DataParallel(decoder)

encoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, encoder.parameters()),
                                   lr = learning_rate)
decoder_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, decoder.parameters()),
                                   lr = learning_rate)

#iter_per_epoch = int(np.ceil(train_size * 1. / BATCH_SIZE))
#iter_losses = np.zeros(n_epochs * iter_per_epoch)
#epoch_losses = np.zeros(n_epochs)

loss_func = nn.MSELoss()

n_iter = 0

encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

input_encoded = encoder(x_batch)
#y_pred = decoder(input_encoded, y_batch).squeeze()
#
#loss = loss_func(y_pred, target_batch)
#loss.backward()
#
#encoder_optimizer.step()
#decoder_optimizer.step()
