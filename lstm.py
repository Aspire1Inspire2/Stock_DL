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
import numpy.random as random
import pickle
import time

torch.cuda.set_device(1)
torch.manual_seed(1)
random.seed(seed=1)
 
# Assign the path the the data pickle file
data_file = './data/Normalized_data/normalized_data_ver5.pickle'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# List hyperparameters here
TRAIN_SIZE = 1
TEST_SIZE = 2
INPUT_DIM = 2
HIDDEN_DIM = 14
PREDICT_DIM = 1
HISTORY_DIM = 180
BATCH_SIZE = 8192

if torch.cuda.is_available():
    MINIBATCH_SIZE = int(BATCH_SIZE / torch.cuda.device_count())
else:
    MINIBATCH_SIZE = BATCH_SIZE        

class StockDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file, tot_entries, history_dim, device):
        """
        Args:
            data_file (string): Path to the data file with stock information.
            history_dim (int): Number of historical records in each sample.
        """
        self.history_dim = history_dim
        self.device = device
        
        data_file_open = open(data_file, 'rb')
        self.stock_data = pickle.load(data_file_open)
        self.stock_index = self.stock_data.index
        self.stock_index_unique = np.unique(self.stock_data.index.values)
        self.data_tag = \
            random.permutation(self.stock_index_unique)[0:tot_entries]
        self.stock_index_counts = self.stock_data.index.value_counts()
        
        self.data_location = []
        self.data_length = 0
        
        for idx in self.data_tag:
            
            start_location = self.stock_index.get_loc(idx).start
            section_length = self.stock_index_counts.loc[idx] \
                             - self.history_dim
            self.data_length = self.data_length + section_length
            
            for i in range( section_length ):
                
                location = start_location + i
                self.data_location.append(location)
                
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        sample = self.stock_data.iloc[self.data_location[idx] : 
            self.data_location[idx] + self.history_dim].values
        targets = self.stock_data['pcnt_diff'].iloc[self.data_location[idx] + 
                                  self.history_dim]

        return (torch.tensor(np.float32(sample), 
                             dtype = torch.float32,
                             requires_grad=False,
                             device = self.device), 
                torch.tensor(np.float32(targets), 
                             dtype = torch.float32,
                             requires_grad=False,
                             device = self.device))


stock_dataset = StockDataset(data_file, 2000, HISTORY_DIM, device)
stock_dataset.__getitem__(1)

stock_dataloader = DataLoader(dataset=stock_dataset, batch_size=BATCH_SIZE, 
                              shuffle=False)

# For testing purposes, we only need to slice some number of samples out of the
# Dataloader. We do not need to run through the whole dataset.
data_iter = stock_dataloader.__iter__()


######################################################################
# Create the model:


class LSTM_Predictor(nn.Module):

    def __init__(self, input_dim, history_dim, hidden_dim, 
                 predict_dim, model_batch_dim, device):
        super(LSTM_Predictor, self).__init__()
        
        self.device = device
        self.history_dim = history_dim
        self.hidden_dim = hidden_dim
        self.model_batch_dim = model_batch_dim

        # The LSTM takes stock price and volume as inputs, and outputs 
        # hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = input_dim, 
                            hidden_size = hidden_dim, 
                            num_layers = 1,
                            batch_first = True)

        # The linear layer that maps from hidden state space to 
        # prediction space
        self.hidden2out = nn.Linear(hidden_dim, predict_dim)

    def init_hidden(self, minibatch_size):
        # Initialize the hidden states. There are two tensors in the tuplet.
        # They are the h_0 state and the C_0 state.
        # The axes semantics are (minibatch_size, num_layers, hidden_dim)
        return [torch.zeros(minibatch_size, 
                            1, 
                            self.hidden_dim,
                            requires_grad=True,
                            device = self.device),
                torch.zeros(minibatch_size, 
                            1, 
                            self.hidden_dim,
                            requires_grad=True,
                            device = self.device)]

    def forward(self, input_data, hidden):
#        print('hidden before', hidden[0].size())
#        print('input', input_data.size())
        for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous()
        
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(input_data, hidden)
        
#        print('output', lstm_out.size())
#        print('hidden after', hidden[0].size())
        prediction = self.hidden2out(lstm_out[0:self.model_batch_dim,
                                              self.history_dim - 1,
                                              0:self.hidden_dim]).squeeze()
#        print('prediction', prediction.size())
        hidden = list(hidden)
        for i in range(len(hidden)):
            hidden[i] = hidden[i].permute(1, 0, 2).contiguous()
            
        return prediction, hidden

######################################################################
# Train the model:


model = LSTM_Predictor(INPUT_DIM, HISTORY_DIM, HIDDEN_DIM, 
                       PREDICT_DIM, MINIBATCH_SIZE, device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model, dim=0)

model.to(device)

# Use the mean square errorloss function to measures the distance of 
# prediction from the actuall stock value
loss_function = nn.MSELoss(size_average=True, reduce=True)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# =============================================================================
data_iter.__init__(stock_dataloader)
with torch.no_grad():
    for item in range(TRAIN_SIZE):
        hidden = model.module.init_hidden(BATCH_SIZE)
        input_data, targets = data_iter.__next__()
#        input_data = input_data.transpose(0,1)
        input_data.to(device)
        targets.to(device)
        tag_scores, hidden = model(input_data, hidden)
        loss = loss_function(tag_scores, targets)
        
        print(loss)
# =============================================================================
start_time = time.time()
for epoch in range(200):
    print(epoch)
    data_iter.__init__(stock_dataloader)
    for item in range(TRAIN_SIZE): #test_tag:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        hidden = model.module.init_hidden(BATCH_SIZE)

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        input_data, targets = data_iter.__next__()
#        input_data = input_data.transpose(0,1)
        input_data.to(device)
        targets.to(device)
        
        # Step 3. Run our forward pass.
        tag_scores, hidden = model(input_data, hidden)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        #print(loss)
        loss.backward()
        optimizer.step()

print("--- %s seconds ---" % (time.time() - start_time))
# See what the scores are after training
# =============================================================================
data_iter.__init__(stock_dataloader)
with torch.no_grad():
    for item in range(TRAIN_SIZE): #test_tag:
        hidden = model.module.init_hidden(BATCH_SIZE)
        input_data, targets = data_iter.__next__()
#        input_data = input_data.transpose(0,1)
        input_data.to(device)
        targets.to(device)
        tag_scores, hidden = model(input_data, hidden)
        loss = loss_function(tag_scores, targets)
        
        print(loss)
# =============================================================================
