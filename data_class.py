# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:12:55 2018

@author: Yuwei Zhu
"""

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

import numpy.random as random
random.seed(seed=1)

class StockDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file, batch_dim, device):
        """
        Args:
            data_file (string): Path to the data file with stock information.
            history_dim (int): Number of historical records in each sample.
        """
        self.batch_dim = batch_dim
        self.device = device
        
        data_file_open = open(data_file, 'rb')
        self.stock_data = pickle.load(data_file_open)
        self.stock_index = self.stock_data.index
        self.stock_index_unique = np.unique(self.stock_index.values)
        self.data_tag = random.permutation(self.stock_index_unique)
        self.stock_index_counts = self.stock_index.value_counts()
        
                
    def __len__(self):
        return len(self.stock_index_unique)

    def __getitem__(self, idx):
        batch_group = int(idx / self.batch_dim)
        location = self.data_tag[batch_group * self.batch_dim : 
            (batch_group + 1) * self.batch_dim]
        max_length = self.stock_index_counts.loc[location].max() - 1
        length = self.stock_index_counts.loc[self.data_tag[idx]] - 1
        pad_length = max_length - length
        sample = self.stock_data.loc[self.data_tag[idx]].iloc[:-1].values
        targets = self.stock_data.loc[self.data_tag[idx],
                                      'pcnt_diff'].iloc[1:].values
        sample = np.pad(sample, [(0, pad_length), (0, 0)], 
                                 mode='constant', constant_values=0)
        targets = np.pad(targets, (0, pad_length), 
                                 mode='constant', constant_values=0)
        
        return (torch.tensor(np.float64(sample), 
                             dtype = torch.float64,
                             requires_grad=False,
                             device = self.device), 
                torch.tensor(np.float64(targets), 
                             dtype = torch.float64,
                             requires_grad=False,
                             device = self.device),
                torch.tensor(np.float64(length),
                             dtype = torch.float64,
                             requires_grad=False,
                             device = self.device))