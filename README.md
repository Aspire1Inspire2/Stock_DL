# Stock_DL

This project is provided by Yuwei Zhu and Chong Shao. It utilizes deep learning
to study the trend of stock data. Dataset is not provided in GitHub due to
storage limitation.

The data is processed by the following methods:
1. Abandon the time tag, since the "absolute time" is not important in time
   series. It should not represent characteristic data in deep learning input.

2. Normalize the volume data by dividing to the standard deviation of each stock

  $ volume = \frac{voclume}{std(volume)} $

   The trade volume for each stock is different. The input to deep learning,
   however, should be homogeneous to study the underlying pattern of stock
   trading behavior.

   I don't know if there is any better way of normalizing the volume. Perhaps
   you might come across with some better ideas?

3. The price is normalized to percentage change since the absolute price is not
   important in the trend, only relative price is important.

   However, one can also normalize the price of an stock to its "oldest" price
   found at the first data entry. In this way, the trend of a stock is
   also maintained.

   I am not sure whether the "percentage change" or the "normalize to oldest" is
   better. Maybe the percentage data gives a better emphasis on the movement
   of stock data. I wonder if they will be equivalent to deep learning or not.

4. Abandon the stock data with too many zeros or missing values in the volume
   data. There are some stocks where the volume data is incomplete to an extend
   that it should be considered junk data.

   I abandoned all the data whose bad data rate > 20%. I set all missing values
   to zero. There are existing zeros in the dataset. I consider them bad data
   points whatsoever.

5. Filled all bad data (zeros and NaN) with predicted distribution and
   reasonable extrapolations.



Normalize_data.py:
It normalize the data in ~/data folder and fill in missing data using predicted
distribution.

sequence_models_tutorial.py:
A tutorial from Pytorch offical website on the LSTM model.
