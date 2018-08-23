import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import argparse
import pickle
import pandas as pd

from DARNN import fintech
from data_class import StockDataset

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
parser.add_argument('--encoder-hsize', default=2, required=False,
                    type=int,
                    help='The number of hidden units')
parser.add_argument('--decoder-hsize', default=16, required=False,
                    type=int,
                    help='The number of hidden units')
parser.add_argument('--pmnist', default=False, action='store_true',
                    help='If set, it uses permutated-MNIST dataset')
parser.add_argument('--batch-size', default=32, required=False, type=int,
                    help='The size of each batch')
parser.add_argument('--n_epochs', default=30, required=False, type=int,
                    help='The maximum iteration count')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='The value specifying whether to use GPU')
parser.add_argument('--parallel', default=False, action='store_true',
                    help='The value specifying whether to use parallel GPU')
parser.add_argument('--test', default=False, action='store_true',
                    help='If run in test mode with a small amount of training data')
args = parser.parse_args()

torch.manual_seed(1)

#Hyperparameter
T = 252 # 252 trading days per year, 126 per half year
y_label = 88331 # The stock PERMNO label,
                #this is the stock to be studied
DATA_PATH = args.data
model_name = args.model
save_dir = args.save
ENCODER_HSIZE = args.encoder_hsize
DECODER_HSIZE = args.decoder_hsize
pmnist = args.pmnist
BATCH_SIZE = args.batch_size
N_EPOCHS = args.n_epochs
USE_GPU = args.gpu
PARALLEL = args.parallel
TEST = args.test

learning_rate = 0.01
TRAIN_SIZE = 5
#TEST_SIZE = 2
INPUT_DIM = 2
#HIDDEN_DIM = 14
PREDICT_DIM = 1
#HISTORY_DIM = 180

device = torch.device("cuda:0" if USE_GPU else "cpu")
if USE_GPU:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

if PARALLEL:
    MINIBATCH_SIZE = int(BATCH_SIZE / torch.cuda.device_count())
else:
    MINIBATCH_SIZE = BATCH_SIZE


# Set the research begin and end date:
begin = pd.to_datetime('2002-12-31')
end = pd.to_datetime('2012-12-31')
time_delta = pd.Timedelta(365, unit='d')


"""
#Load the original data
stock_data = pickle.load(open(DATA_PATH, 'rb')).reset_index(level = 1)

stat_eval = stock_data.loc[begin - time_delta : end, 
                           ['PERMNO', 'RET', 'VOL']].copy()

stat_eval = stat_eval.pivot(columns='PERMNO', values=['RET', 'VOL'])

ret_eval = stat_eval['RET']
vol_eval = stat_eval['VOL']

rolling = vol_eval.rolling(T)
vol_mean = rolling.mean().loc[begin : end]
#stat_min = rolling.min().loc[begin : end]
#stat_max = rolling.max().loc[begin : end]
vol_std = rolling.std().loc[begin : end]

rolling = ret_eval.rolling(T)
ret_std = rolling.std().loc[begin : end]

# Locate the input data to learn
input_data = stock_data.loc[begin : end].copy()

# From the semi-one-dimension table create a two dim table
# The VOL data is normalized
# This two dimension table is used in the Pytorch dataset as input
input_data = input_data.pivot(columns='PERMNO', values=['RET','VOL'])
input_data['VOL'] = (input_data['VOL'] - vol_mean) / vol_std
input_data['RET'] = input_data['RET'] / ret_std
input_data = input_data.swaplevel(axis=1).sort_index(axis=1)

with open('data/input.pickle', 'wb') as handle:
    pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
"""

# Load the processed data to accerlate debugging
# Before using this line, use the above block to dump the data.
input_data = pickle.load(open('data/input.pickle', 'rb'))

"""
#
## Use the following block to select only part of the whole input data
#columns = np.unique(input_data.columns.get_level_values(0).values)
#trading_value = pickle.load(open('data/trading_value.pickle', 'rb'))
#trading_value = trading_value.loc[columns] \
#                       .sort_values('AMNT', ascending = False)
#ranking = trading_value.index \
#                       .get_level_values(0) \
#                       .values
#cutoff = (trading_value.values.squeeze() / trading_value.iloc[0].values) > 1e-3
#keep = ranking[cutoff]
#input_data = input_data[keep]
"""

train = input_data.loc[begin : end - time_delta]
test = input_data.loc[end - time_delta * 2 : end]

train_dataset = StockDataset(train, T, y_label, device)
test_dataset = StockDataset(test, T, y_label, device)

# Assign the Dataloader to automatically load batched data for you
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last = True)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             drop_last = True)

n_stock = int(input_data.shape[1] / 2 - 1)

# Assign the model
fintech_model = fintech(n_stock = n_stock,
                  batch_size = MINIBATCH_SIZE,
                  encoder_hidden_size = ENCODER_HSIZE, 
                  decoder_hidden_size = DECODER_HSIZE,
                  T = T,
                  device = device)

# Load the model to multiple GPU
if PARALLEL:
    fintech_model = nn.DataParallel(fintech_model, dim=0)

fintech_model.to(device)

# Assign optimizer
fintech_model_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad, 
                                               fintech_model.parameters()),
                               lr = learning_rate,
                               weight_decay = 0.01)

# Assign loss function
loss_func = nn.BCEWithLogitsLoss()

# Lets try the data loader
if TEST:
    data_iter = train_dataloader.__iter__()

# Train the data
for n_iter in range(N_EPOCHS):
    print('Epoch:', n_iter)
    loss_epoch = []

    if TEST:
        data_iter.__init__(train_dataloader)

        for i in range(TRAIN_SIZE):
            x_batch, y_batch, target_batch = data_iter.__next__()
            fintech_model.zero_grad()

            y_pred = fintech_model(x_batch, y_batch)

            loss = loss_func(y_pred, target_batch)
            #print(loss.item())
            loss_epoch.append(loss.item())
            loss.backward()

            fintech_model_optimizer.step()
    else:
        for x_batch, y_batch, target_batch in train_dataloader:
            fintech_model.zero_grad()

            y_pred = fintech_model(x_batch, y_batch)

            loss = loss_func(y_pred, target_batch)
            #print(loss.item())
            loss_epoch.append(loss.item())
            loss.backward()

            fintech_model_optimizer.step()

#    print('Epoch loss:', loss_epoch)
    print('Average epoch loss:', sum(loss_epoch)/len(loss_epoch))

    # Back testing the data
    with torch.no_grad():
        test_epoch = []
        for x_batch, y_batch, target_batch in test_dataloader:
            y_pred = fintech_model(x_batch, y_batch)
            loss = loss_func(y_pred, target_batch)
            test_epoch.append(loss.item())
#        print('Test loss:', test_epoch)
        print('Average test loss:', sum(test_epoch)/len(test_epoch))


torch.save(fintech_model, save_dir + '/fintech_model')
