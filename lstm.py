"""
Please refer the basic model to this link:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
This script applies the basic LSTM algorithm in Pytorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import random
import pickle

torch.cuda.set_device(1)
torch.manual_seed(1)
random.seed(1)

# Assign the path the the data pickle file
data_file = './data/Normalized_data/normalized_data_ver5.pickle'
HISTORY_DIM = 180
BATCH_SIZE = 10

class StockDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file, history_dim):
        """
        Args:
            data_file (string): Path to the data file with stock information.
            history_dim (int): Number of historical records in each sample.
        """
        
        data_file_open = open(data_file, 'rb')
        self.stock_data = pickle.load(data_file_open)
        self.stock_index = self.stock_data.index
        self.stock_index_unique = np.unique(self.stock_data.index.values)
        self.stock_index_counts = self.stock_data.index.value_counts()
        self.history_dim = history_dim        
        
        self.data_tag = []
        self.data_length = 0
        
        for idx in self.stock_index_unique:
            
            start_location = self.stock_index.get_loc(idx).start
            section_length = int( self.stock_index_counts.loc[idx] / 
                                 self.history_dim)
            self.data_length = self.data_length + section_length
            
            for i in range( section_length ):
                
                location = start_location + self.history_dim * i
                self.data_tag.append(location)
                
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        sample = self.stock_data.iloc[self.data_tag[idx] : 
            self.data_tag[idx] + self.history_dim].values
        targets = self.stock_data['pcnt_diff'].iloc[self.data_tag[idx] + 1 : 
            self.data_tag[idx] + self.history_dim + 1].values

        return sample, targets

stock_dataset = StockDataset(data_file, HISTORY_DIM)


stock_dataloader = DataLoader(dataset=stock_dataset, batch_size=BATCH_SIZE, 
                              shuffle=False)

# For testing purposes, we only need to slice some number of samples out of the
# Dataloader. We do not need to run through the whole dataset.
data_iter = stock_dataloader.__iter__()
data_iter.__init__(stock_dataloader)
for i in range(2):
    sample, targets = data_iter.__next__()
    print(sample)
    

# List hyperparameters here
TRAIN_SIZE = 100
TEST_SIZE = 20
INPUT_DIM = 2
HIDDEN_DIM = 14
PREDICT_DIM = 1

######################################################################
# Create the model:


class LSTM_Predictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, predict_dim, batch_dim):
        super(LSTM_Predictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_dim = batch_dim

        # The LSTM takes stock price and volume as inputs, and outputs 
        # hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first = True)

        # The linear layer that maps from hidden state space to prediction space
        self.hidden2out = nn.Linear(hidden_dim, predict_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Initialize the hidden states. There are two tensors in the tuplet.
        # They are the h_0 state and the C_0 state.
        # The axes semantics are (minibatch_size, num_layers, hidden_dim)
        return (torch.zeros(self.batch_dim, 1, self.hidden_dim).cuda(1),
                torch.zeros(self.batch_dim, 1, self.hidden_dim).cuda(1))

    def forward(self, input_data):
        lstm_out, self.hidden = self.lstm(input_data, self.hidden)
        prediction = self.hidden2out(lstm_out.view(len(input_data), -1)).squeeze()
        #tag_scores = F.log_softmax(predict_space, dim=1)
        return prediction

######################################################################
# Train the model:


model = LSTM_Predictor(INPUT_DIM, HIDDEN_DIM, PREDICT_DIM, BATCH_SIZE)
model.cuda(1)
#model = model.double()

# Use the mean square errorloss function to measures the distance of 
# prediction from the actuall stock value
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# =============================================================================
data_iter.__init__(stock_dataloader)
with torch.no_grad():
    for item in range(TRAIN_SIZE): #test_tag:
        input_data, targets = data_iter.__next__()
        tag_scores = model(input_data)
        loss = loss_function(tag_scores, targets)
        
        print(loss)
# =============================================================================
for epoch in range(300):
    print(epoch)
    data_iter.__init__(stock_dataloader)
    for item in range(TRAIN_SIZE): #test_tag:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        input_data, targets = data_iter.__next__()

        # Step 3. Run our forward pass.
        tag_scores = model(input_data)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        #print(loss)
        loss.backward()
        optimizer.step()

# See what the scores are after training
# =============================================================================
with torch.no_grad():
    for tag in train_tag: #test_tag:
        input_data = load_data(data, tag, HISTORY_DIM)
        targets = torch.tensor(
               data['pcnt_diff'].iloc[tag + 1 : tag + (HISTORY_DIM + 1)].values,
                                        dtype = torch.float).cuda(1)
        tag_scores = model(input_data)
        loss = loss_function(tag_scores, targets)
        
        print(loss)
# =============================================================================
