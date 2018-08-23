# A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
# source: https://arxiv.org/pdf/1704.02971.pdf

import torch
from torch import nn
import torch.nn.functional as F


class fintech(nn.Module):
    def __init__(self, n_stock, batch_size, encoder_hidden_size, 
                 decoder_hidden_size, T, device):
        """
        n_stock: number of stocks
        batch_size: batch size
        hidden_size: hidden size
        T: number of time steps
        device: running device
        """
        super(fintech, self).__init__()
        self.n_stock = n_stock
        self.n_fea = n_stock * 2 # number of features
        self.batch_size = batch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.T = T
        self.device = device
        
        self.attn_weights = nn.Parameter(torch.rand([batch_size, n_stock],
                            dtype=None, device=device, requires_grad=False))
        
        self.weight = nn.Linear(self.n_fea, encoder_hidden_size)
        self.lstm_encode = nn.LSTM(input_size = encoder_hidden_size + 2,
                                   hidden_size = 64,
                                   bidirectional = False,
                                   batch_first = True)
        self.final = nn.Linear(64, 1)
#        nn.Sequential(
#                        nn.Linear(encoder_hidden_size + 2, decoder_hidden_size),
#                        nn.ReLU(),
#                        nn.Linear(decoder_hidden_size, 1)
#                        )
        for i in range(n_stock):
            setattr(self, 'encoder_{}'.format(i), 
                    nn.LSTM(input_size = 2,
                            hidden_size = encoder_hidden_size,
                            bidirectional = False,
                            batch_first = True))
        
        
        
        
        
    def forward(self, input_data, y_history):
        alpha = F.softmax(self.compress(self.attn_weights),
                          dim = 1) * self.n_stock
        alpha = alpha.transpose(0,1) \
                     .repeat(1, 2) \
                     .view(self.n_fea, self.batch_size) \
                     .transpose(0,1) \
                     .unsqueeze(1) \
                     .repeat(1, self.T, 1)
        output_lstm = {}
        for i in range(self.n_stock):
            input_i = input_data[:, :, (i-1)*2 : (i*2)]
            _, (output_lstm[i], _) = \
            getattr(self, 'encoder_{}'.format(i))(input_i)
            
        
        
        input_weighted = self.weight(input_data)
        input_cat = torch.cat((input_weighted, y_history), dim = 2)
        
        _, (exogenous_encoded, _) = self.lstm_encode(input_cat)
        y_pred = self.final(exogenous_encoded.squeeze(dim = 0))
        
        return y_pred.squeeze()