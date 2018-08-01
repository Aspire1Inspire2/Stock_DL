# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:26:52 2018

@author: Yuwei Zhu
"""
import torch.nn as nn

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
        
        # This flatten command cause error under multiGPU mode.
        #self.lstm.flatten_parameters()
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