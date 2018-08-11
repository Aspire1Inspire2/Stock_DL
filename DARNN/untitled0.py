# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:00:06 2018

@author: Yuwei Zhu
"""
import pickle

data_file_open = open('normalized_data_6ver.pickle', 'rb')

stock_data = pickle.load(data_file_open)

stock_data = stock_data.swaplevel().sort_index()

a = stock_data.loc['2012-12-31'].index.get_level_values(1)
b = stock_data.loc['2002-12-31'].index.get_level_values(1)
c = a.intersection(b)

