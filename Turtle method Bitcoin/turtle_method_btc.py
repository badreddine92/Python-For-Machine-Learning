import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bitcoin = pd.read_csv('BTC-EUR.csv',index_col='Date', parse_dates=True)

data = bitcoin.copy()
data['Buy']=np.zeros(len(data))
data['Sell']=np.zeros(len(data))
data['rollingMax'] = data['Close'].shift(1).rolling(window=28).max()
data['rollingMin'] = data['Close'].shift(1).rolling(window=28).min()
data.loc[data['rollingMax'] < data['Close'] , 'Buy'] = 1
data.loc[data['rollingMin'] > data['Close'] , 'Sell'] = -1

start = '2018'
end = '2023'

fig, ax = plt.subplots(2,figsize=(6,6),sharex=True,)
plt.title(label='Turtle method on Bitcoin since 2018')
ax[0].plot(data['Close'][start:end])
ax[0].plot(data['rollingMax'][start:end],color ='green')
ax[0].plot(data['rollingMin'][start:end], color='red')
ax[1].plot(data['Buy'][start:end], color='red')
ax[1].plot(data['Sell'][start:end], color='green')
plt.show()