import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#Hyperparameter
T = 252 # 252 trading days per year
BATCH_SIZE = 2 # test batch size
y_label = 10001

# Assign the path the the Pandas data file here
data_file_open = open('data/python_stock_data.pickle', 'rb')

#Load the original data
stock_data = pickle.load(data_file_open)

# Set the research begin and end date:
temp = stock_data.loc['2002-12-31':'2012-12-31'].reset_index()

# From the semi-one-dimension table create a two dim table
# This two dimension table is used in the Pytorch dataset as input
temp = temp.pivot(index='DATE', columns='PERMNO', values=['RET','VOL'])
temp = temp.swaplevel(axis=1).sort_index(axis=1)


# Ignore the following lines, these are some backup lines

# stock_byid = stock_data.swaplevel().sort_index()

# unique_column = np.unique(temp.index.get_level_values(1).values)
# unique_row = np.unique(temp.index.get_level_values(0).values)

# column_index = pd.MultiIndex.from_product([unique_column,['VOL','RET']])
# row_index = pd.Index(unique_row, name='date')


# Define the Dataset class, to be later used in Pytorch Dataloader
class StockDataset(Dataset):
    """
    Stock dataset.
    """

    def __init__(self, pd_data, T, ylabel, device):
        """
        Args:
            pd_data (pandas DataFrame): The data.
            T (int): Number of historical records in each learning sample.
            device: cuda device or cpu.
        """
        self.T = T
        self.ylabel = ylabel
        self.device = device

        self.stock_data = pd_data
        self.total_time = len(self.stock_data)
        self.tot_num = len(np.unique(pd_data.columns.get_level_values(0).values))

    def __len__(self):
        return self.total_time - self.T + 1

    def __getitem__(self, idx):
        """
        Return a tuple,
        the first result is the prediction input data
        the second result is the larning target value

        The input data is pytorch tensor with the following shape:
        (number of stocks in the sample, number of timesteps, 2)
        In the last dimension, there are two numbers, 
        the first number is price change ('RET')
        the second number is VOLumn ('VOL')
        """
        begin = idx
        end = idx + self.T
        
        x = self.stock_data.drop(columns=self.ylabel).iloc[begin:end]
        x = x.swaplevel(axis=1)
        x['VOL'] = (x['VOL'] - x['VOL'].mean()) / (x['VOL'].max() - x['VOL'].min())
        x = x.fillna(0).values
        x = x.reshape((self.T, self.tot_num - 1, 2)).transpose(1, 0, 2)
        x = torch.tensor(np.float64(x), 
                          dtype = torch.float64,
                          requires_grad=False,
                          device = self.device)
    
        y = self.stock_data[self.ylabel].iloc[begin:end]
        y['VOL'] = (y['VOL'] - y['VOL'].mean()) / (y['VOL'].max() - y['VOL'].min())
        y = y.fillna(0).values
        y = torch.tensor(np.float64(y), 
                         dtype = torch.float64,
                         requires_grad=False,
                         device = self.device)
        
        target = self.stock_data[self.ylabel].iloc[end].fillna(0).values[0]
        #target = targets.reshape((self.tot_num,2)).transpose()[0]
        target =  torch.tensor(np.float64(target), 
                               dtype = torch.float64,
                               requires_grad=False,
                               device = self.device)
        
        return x, y, target

# Here is how to use it:
# 1. Put the processed two dim table
# 2. Put the number of trading days in lstm input
# 3. Put the device name
stock_dataset = StockDataset(temp, T, y_label, 'cpu')

# Let's try out the first sample
# Compare to temp[10001].iloc[0:253] to see it successfully loaded
x, y, target = stock_dataset.__getitem__(0)
print(x.size())
print(y.size())
print(target.size())

# Assign the Dataloader to automatically load batched data for you
stock_dataloader = DataLoader(dataset=stock_dataset, batch_size=BATCH_SIZE,
                              shuffle=False)

# Lets try out the dataloader
data_iter = stock_dataloader.__iter__()
data_iter.__init__(stock_dataloader)
x_batch, y_batch, target_batch = data_iter.__next__()
