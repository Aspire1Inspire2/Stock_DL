import os
import numpy as np
import pandas as pd
import pickle


# To run this data processing code, put the raw data under data folder in the 
# working directory
root_dir = '.'

data_dir = root_dir + '/data/'
file_list = os.listdir(data_dir)

stock = pd.DataFrame()

for i in file_list:
    if (i[2:] == '000'):
        print('Reading:',i)
        
    temp = pd.read_csv(data_dir + i, delim_whitespace = True, 
                       header = None, usecols=[1,2], names = ['prc','vol'])
    
    temp = temp.replace({0:np.nan})
    if (temp['vol'].isnull().sum() / len(temp) > 0.2 or len(temp) < 185):
        continue
    else:
        temp['prc'] = temp['prc'].abs().interpolate(limit_direction='both')
        temp['pcnt_diff'] = temp['prc'].diff(periods=1)
        temp['pcnt_diff'] = temp['pcnt_diff'] / temp['prc'] * 100
        temp = temp.drop(columns=['prc'])
        
        #tmp = temp[temp['vol'].notnull()]
        temp['vol'] = temp['vol'] / temp['vol'][temp['vol'].notnull()].std()
        temp['id'] = int(i)
        stock = stock.append(temp)
        
stock = stock.set_index('id')


# Deal with missing data points
stock["pcnt_diff"] = stock["pcnt_diff"].fillna(0)
stock["pcnt_diff"][stock["pcnt_diff"] < -99.99] = 0
stock_no_zeros = stock[stock['vol'].notnull()]

bins = pd.cut(stock_no_zeros["pcnt_diff"], np.arange(-100, 101, 1))
stock_grouped = stock_no_zeros['vol'].groupby(bins)
vol_dist = stock_grouped.agg(['mean', 'count','std'])

stock_tag = pd.cut(stock['pcnt_diff'], np.arange(-100, 101, 1))
temp = vol_dist['mean'].loc[stock_tag[stock['vol'].isnull()]]
stock['vol'][stock['vol'].isnull()] = temp.values

with open('normalized_data_4ver.pickle', 'wb') as handle:
    pickle.dump(stock, handle, protocol=pickle.HIGHEST_PROTOCOL)