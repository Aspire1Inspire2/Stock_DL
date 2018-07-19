"""
Please refer the basic model to this link:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
This script applies the basic LSTM algorithm in Pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import random
import pickle

torch.manual_seed(1)
random.seed(1)

# Assign the path the the data pickle file
data_file = open('./data/Normalized_data/normalized_data_ver5.pickle', 'rb')
data = pickle.load(data_file)


def load_data(data, location, length):
    """
    Load the learning data
    data is the input Pandas DataFrame, 
    location is the position of the data,
    length is the number of historical entries in the input to the lstm model
    """
    return torch.tensor(data.iloc[location : location + length].values, 
                        dtype=torch.float)

def build_dataset(data, history_dim, train_size, test_size):
    """
    Build the training and testing dataset. The entries in these two dataset
    are non-repetive. Realized by calling the random.sample method
    data is the input Pandas DataFrame, 
    history_dim is the number of historical entries to learn
    train_size is the training dataset size
    test_size is the testing dataset size
    """
    np.random.seed(seed=1)
    if train_size + test_size > len(data):
        raise Exception("Training and testing size larger than the total" + 
                        " number of available data.")
    else:
        data_tag = np.arange(len(data) - (history_dim + 1) )
        np.random.shuffle(data_tag)
        
        train_tag = []
        test_tag = []
        
        i = 0
        for location in data_tag:
            if train_size < i: break
            if (data.iloc[location].name == 
            data.iloc[location + history_dim].name):
                i = i + 1
                train_tag.append(location)
        
        i = 0
        for location in np.flip(data_tag, axis = 0):
            if test_size < i: break
            if (data.iloc[location].name == 
            data.iloc[location + (history_dim + 1)].name):
                i = i + 1
                test_tag.append(location)
    return train_tag, test_tag
    

# List hyperparameters here
HISTORY_DIM = 180
TRAIN_SIZE = 100
TEST_SIZE = 20
INPUT_DIM = 2
HIDDEN_DIM = 14
PREDICT_DIM = 2

######################################################################
# Create the model:


class LSTM_Predictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, predict_dim):
        super(LSTM_Predictor, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes stock price and volume as inputs, and outputs 
        # hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to prediction space
        self.hidden2out = nn.Linear(hidden_dim, predict_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Initialize the hidden states. There are two tensors in the tuplet.
        # They are the h_0 state and the C_0 state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, stock_data):
        lstm_out, self.hidden = self.lstm(
            stock_data.view(len(stock_data), 1, -1), self.hidden)
        prediction = self.hidden2out(lstm_out.view(len(stock_data), -1))
        #tag_scores = F.log_softmax(predict_space, dim=1)
        return prediction

######################################################################
# Train the model:

train_tag, test_tag = build_dataset(data, HISTORY_DIM, TRAIN_SIZE, TEST_SIZE)

model = LSTM_Predictor(INPUT_DIM, HIDDEN_DIM, PREDICT_DIM)
#model = model.double()

# Use the mean square errorloss function to measures the distance of 
# prediction from the actuall stock value
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
#with torch.no_grad():
#    inputs = prepare_sequence(training_data[0][0], word_to_ix)
#    tag_scores = model(inputs)
#    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for tag in train_tag:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        input_data = load_data(data, tag, HISTORY_DIM)
        targets = torch.tensor(
                data.iloc[tag + 1 : tag + (HISTORY_DIM + 1)].values,
                                         dtype = torch.float)

        # Step 3. Run our forward pass.
        tag_scores = model(input_data)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        print(loss)
        loss.backward()
        optimizer.step()

# See what the scores are after training
# =============================================================================
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
# 
#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(tag_scores)
# =============================================================================
