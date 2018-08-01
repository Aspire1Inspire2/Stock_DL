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