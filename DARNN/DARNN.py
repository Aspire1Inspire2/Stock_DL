# A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
# source: https://arxiv.org/pdf/1704.02971.pdf

import torch
from torch import nn
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, n_stock, batch_size, hidden_size, T, device):
        """
        n_stock: number of stocks
        batch_size: batch size
        hidden_size: hidden size
        T: number of time steps
        device: running device
        """
        super(encoder, self).__init__()
        self.n_stock = n_stock
        self.n_fea = n_stock * 2 # number of features
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.T = T
        self.device = device


        self.attn_weights = nn.Parameter(torch.rand([batch_size, n_stock],
                            dtype=None, device=device, requires_grad=False))

        self.y_encode = nn.LSTM(input_size = 2,
                                   hidden_size = hidden_size,
                                   batch_first = True)
        self.exogenous_encode = nn.LSTM(input_size = self.n_fea,
                                        hidden_size = hidden_size,
                                        batch_first = True)
        self.compress = nn.Tanh()
    # TODO(chongshao): Maybe find a specific large matrix multiply function?
    def forward(self, input_data, y_history):
#        print('alpha max', self.attn_weights.max())
#        print('alpha min', self.attn_weights.min())
        alpha = F.softmax(self.compress(self.attn_weights),
                          dim = 1) * self.n_stock
        alpha = alpha.transpose(0,1) \
                     .repeat(1, 2) \
                     .view(self.n_fea, self.batch_size) \
                     .transpose(0,1) \
                     .unsqueeze(1) \
                     .repeat(1, self.T, 1)
        weighted_input = torch.mul(alpha, input_data)

        y_encoded, (_, _) = self.y_encode(y_history)

        exogenous_encoded, (_, _) = self.exogenous_encode(weighted_input)

        return exogenous_encoded, y_encoded



class decoder(nn.Module):
    def __init__(self, batch_size, encoder_hidden_size,
                 decoder_hidden_size, T, device):
        super(decoder, self).__init__()

        self.batch_size = batch_size
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.device = device

        self.attn_layer = nn.Parameter(torch.rand([batch_size, T],
                            dtype=None, device=device, requires_grad=False))
        self.lstm_decode = nn.LSTM(input_size = encoder_hidden_size * 2,
                                   hidden_size = decoder_hidden_size,
                                   bidirectional = True,
                                   batch_first = True)
        self.final = nn.Sequential(
                nn.Linear(decoder_hidden_size * 2, decoder_hidden_size),
                nn.Linear(decoder_hidden_size, 1),
                nn.Softmax()
                )
        self.compress = nn.Tanh()

    def forward(self, exogenous_encoded, y_encoded):
#        print('beta max', self.attn_layer.max())
#        print('beta min', self.attn_layer.min())
        beta = F.softmax(self.compress(self.attn_layer),
                         dim = 1) * self.T
        beta = beta.unsqueeze(dim=2) \
                   .repeat(1, 1, self.encoder_hidden_size * 2)

        y_tilde = torch.cat((exogenous_encoded, y_encoded), dim=2)
        context = torch.mul(beta, y_tilde)

        _, (hn, _) = self.lstm_decode(context)

        hn = hn.transpose(0, 1).contiguous()
        hn = hn.view(self.batch_size, self.decoder_hidden_size * 2)
        y_pred = self.final(hn).squeeze()

        return y_pred
