# A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
# source: https://arxiv.org/pdf/1704.02971.pdf

import torch
from torch import nn
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, n_stock, batch_size, hidden_size, T, device):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.n_stock = n_stock
        self.n_fea = n_stock * 2 # number of features
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.T = T
        self.device = device
        
        # hidden, cell: initial states with dimention hidden_size
        self.hidden = torch.rand([batch_size, hidden_size],
                            dtype=None, device=device, requires_grad=False)
        self.cell = torch.rand([batch_size, hidden_size],
                            dtype=None, device=device, requires_grad=False)
        self.attn_weights = nn.Parameter(torch.rand([batch_size, n_stock],
                            dtype=None, device=device, requires_grad=False))
        # perceptron in the paper is taking too much memory
#        self.perceptron = nn.Sequential(
#            nn.Linear(in_features = T + 2 * hidden_size, out_features = T),
#            nn.Tanh()
#            nn.Linear(in_features = T, out_features = 1)
#        )
        
        self.nto1 = nn.Linear(in_features = T + 2 * hidden_size, 
                                      out_features = 1)
        self.attn_linear = nn.Linear(in_features = 2, out_features = 1)
        self.lstm = nn.LSTMCell(input_size = self.n_fea, hidden_size = hidden_size)
        

    def forward(self, input_data):
        # input_data: batch_size * T - 1 * n_stock
        # input_weighted = torch.zeros([self.batch_size, self.T, self.n_fea],
        #                               dtype=None, device=self.device, requires_grad=False)
        input_encoded = torch.zeros([self.batch_size, self.T, self.hidden_size],
                                     dtype=None, device=self.device, requires_grad=False)
        attn_weights = self.attn_weights.transpose(0,1) \
                                        .repeat(1, 2) \
                                        .view(self.n_fea, self.batch_size) \
                                        .transpose(0,1)
        
        for t in range(self.T):
            print(t)
            #Eqn. 2
            # (h0, c0) = (self.hidden, self.cell)
            # h1, c1 =  self.lstm(input_data[:, t, :], (h0, c0))
            
            # Eqn. 8: concatenate the hidden states with each predictor
            #x = torch.cat((self.hidden.unsqueeze(dim=1) \
            #                 .repeat(1, self.n_fea, 1),
            #               self.cell.unsqueeze(dim=1) \
            #                 .repeat(1, self.n_fea, 1),
            #               input_data.transpose(1,2)), dim = 2)
            #
            # Perceptron in the paper is taking too much memory,
            # maybe the line below would be helpful
            # e_t = self.perceptron(x).squeeze(dim=2)
            #
            #e_t = self.nto1(x).squeeze(dim=2)
            #e_t = self.attn_linear(e_t.view(self.batch_size, self.n_stock, 2))
            #e_t = e_t.squeeze(dim=2)
            
            # Eqn. 9: Get attention weights
            # batch_size * n_stock, attn weights with values sum up to n_stock
            #attn_weights = F.softmax(e_t, dim = 1) * self.n_stock 
            
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * n_stock
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            #self.lstm.flatten_parameters()
            
            # Eqn. 11
            (self.hidden, self.cell) = self.lstm(weighted_input, (self.hidden, self.cell))
            # Save output
            # input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = self.hidden
        return input_encoded



class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T - 1 * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T - 1), dim = 1) # batch_size * T - 1, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size
            if t < self.T - 1:
                # Eqn. 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))
        return y_pred

    def init_hidden(self, x):
        return torch.zeros([1, x.size(0), self.decoder_hidden_size],
                            dtype=None, device=None, requires_grad=False)