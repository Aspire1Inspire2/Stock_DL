import numpy as np
import torch
from torch.utils.data import Dataset

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
        the first number is price change (Return 'RET')
        the second number is Volumn ('VOL')
        """
        begin = idx
        end = idx + self.T
        
        x = self.stock_data.drop(columns=self.ylabel).iloc[begin:end]
        x = x.swaplevel(axis=1)
        # normalize data
        x['VOL'] = (x['VOL'] - x['VOL'].mean()) / (x['VOL'].max() - x['VOL'].min())
        x['RET'] = (x['RET'] - x['RET'].mean()) / (x['RET'].max() - x['RET'].min())
        x = x.fillna(0).values
        #x = x.reshape((self.T, self.tot_num - 1, 2)).transpose(1, 0, 2)
        x = torch.tensor(np.float64(x), 
                          dtype = torch.float64,
                          requires_grad=False,
                          device = self.device)
        
        y = self.stock_data[self.ylabel].iloc[begin:end]
        target = self.stock_data[self.ylabel].iloc[end].fillna(0).values[0]
        
        #normalize data
        y['VOL'] = (y['VOL'] - y['VOL'].mean()) / (y['VOL'].max() - y['VOL'].min())
        target = (target - y['RET'].mean()) / (y['RET'].max() - y['RET'].min())
        y['RET'] = (y['RET'] - y['RET'].mean()) / (y['RET'].max() - y['RET'].min())
        
        y = y.fillna(0).values
        y = torch.tensor(np.float64(y), 
                         dtype = torch.float64,
                         requires_grad=False,
                         device = self.device)
        
        #target = targets.reshape((self.tot_num,2)).transpose()[0]
        target =  torch.tensor(np.float64(target), 
                               dtype = torch.float64,
                               requires_grad=False,
                               device = self.device)
        
        return x, y, target