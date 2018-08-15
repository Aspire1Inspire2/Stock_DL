import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#Hyperparameter
T = 252 # 252 trading days per year
BATCH_SIZE = 2 # test batch size


# Assign the path the the Pandas data file here
data_file_open = open('data/normalized_data_6ver.pickle', 'rb')

#Load the original data
stock_data = pickle.load(data_file_open)

# Set the research begin and end date:
temp = stock_data.loc['2002-12-31':'2012-12-31']

# From the semi-one-dimension table create a two dim table
# This two dimension table is used in the Pytorch dataset as input
temp = temp.reset_index().pivot(index='date', columns='id', values=['prc_diff','vol'])
temp = temp.swaplevel(axis=1).sort_index(axis=1)


# Ignore the following lines, these are some backup lines

# stock_byid = stock_data.swaplevel().sort_index()

# unique_column = np.unique(temp.index.get_level_values(1).values)
# unique_row = np.unique(temp.index.get_level_values(0).values)

# column_index = pd.MultiIndex.from_product([unique_column,['vol','prc_diff']])
# row_index = pd.Index(unique_row, name='date')


# Define the Dataset class, to be later used in Pytorch Dataloader
class StockDataset(Dataset):
    """
    Stock dataset.
    """

    def __init__(self, pd_data, T, device):
        """
        Args:
            pd_data (pandas DataFrame): The data.
            T (int): Number of historical records in each learning sample.
            device: cuda device or cpu.
        """
        self.T = T
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
        the first number is price change ('prc_diff')
        the second number is volumn ('vol')
        """
        begin = idx
        end = idx + self.T

        sample = self.stock_data.iloc[begin:end].values
        sample = sample.reshape((self.T,self.tot_num,2)).transpose(1,0,2)

        targets = self.stock_data.iloc[end].values
        targets = targets.reshape((self.tot_num,2)).transpose()[0]

        return (torch.tensor(np.float64(sample),
                             dtype = torch.float64,
                             requires_grad=False,
                             device = self.device),
                torch.tensor(np.float64(targets),
                             dtype = torch.float64,
                             requires_grad=False,
                             device = self.device))

# Here is how to use it:
# 1. Put the processed two dim table
# 2. Put the number of trading days in lstm input
# 3. Put the device name
stock_dataset = StockDataset(temp, T, 'cpu')

# Let's try out the first sample
# Compare to temp[10001].iloc[0:253] to see it successfully loaded
sample, targets = stock_dataset.__getitem__(0)
print(sample.shape)
print(targets.shape)

# Assign the Dataloader to automatically load batched data for you
stock_dataloader = DataLoader(dataset=stock_dataset, batch_size=BATCH_SIZE,
                              shuffle=False)

# Lets try out the dataloader
data_iter = stock_dataloader.__iter__()
data_iter.__init__(stock_dataloader)
sample, targets = data_iter.__next__()
