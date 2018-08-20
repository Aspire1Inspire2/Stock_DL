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

        self.x = pd_data.drop(columns=ylabel).copy().fillna(0)
        self.y = pd_data[ylabel].copy().fillna(0)
        self.target = pd_data[ylabel, 'RET'].copy().fillna(0)
        self.total_time = len(pd_data)
        
    def __len__(self):
        return self.total_time - self.T

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
        end = idx + self.T
        
        x = self.x.iloc[idx:end].values
        x = torch.tensor(x, dtype = torch.float64,
                          requires_grad=False,
                          device = self.device)
        
        y = self.y.iloc[idx:end].values
        y = torch.tensor(y, dtype = torch.float64,
                         requires_grad=False,
                         device = self.device)
        
        
        target = self.target.iloc[end]
        target =  torch.tensor(target, 
                               dtype = torch.float64,
                               requires_grad=False,
                               device = self.device)
        
        return x, y, target