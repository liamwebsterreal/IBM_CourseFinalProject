
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats.mstats import normaltest
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

warnings.simplefilter("ignore")

#%%
filepath_crypto = "Data/CryptoData/coin_Bitcoin.csv"
filepath_electricity = "Data/MER_T09_08.csv"

data_crypto = pd.read_csv(filepath_crypto)
data_electricity = pd.read_csv(filepath_electricity)

#%%
# Data Cleaning

# Getting rid of unnecessary columns
data_crypto_clean = data_crypto.drop(['SNo','Name', 'Symbol', 'Open', 'Close', 'Volume', 'Marketcap'], axis=1)
data_electricity_clean = data_electricity.drop(['Column_Order', 'Unit'], axis=1)
# Changing data types to numeric
data_crypto_clean['Date'] = pd.to_datetime(data_crypto_clean['Date'])
data_crypto_clean['Date'] = data_crypto_clean['Date'].dt.date
data_crypto_clean['Date'] = data_crypto_clean['Date'].astype(str).str.replace('-','')
data_crypto_clean['Date'] = data_crypto_clean['Date'].astype(int)
data_crypto_clean['Date'] = data_crypto_clean['Date'].floordiv(100)
data_crypto_clean.rename(columns={'Date':'YYYYMM'}, inplace=True)
data_crypto_clean.drop(data_crypto_clean[(data_crypto_clean.YYYYMM < 201305) | (data_crypto_clean.YYYYMM > 202106)].index, inplace=True)
data_electricity_clean.drop(data_electricity_clean[(data_electricity_clean.YYYYMM < 201305) | (data_electricity_clean.YYYYMM > 202106)].index, inplace=True)
data_electricity_clean.drop(data_electricity_clean[(data_electricity_clean.MSN == 'ESOTUUS') | (data_electricity_clean.MSN == 'ESRCUUS')].index, inplace=True)
data_electricity_clean.drop(data_electricity_clean[data_electricity_clean.Description != 'Average Price of Electricity to Ultimate Customers, Total'].index, inplace=True)
data_electricity_clean.drop('MSN', axis=1, inplace=True)
data_electricity_clean.drop('Description', axis=1, inplace=True)
data_electricity_clean['Value'] = pd.to_numeric(data_electricity_clean['Value'], downcast='float')
# Finding mean of high and low daily crypto price
data_crypto_clean['Value'] = (data_crypto_clean['High'] + data_crypto_clean['Low']) / 2
data_crypto_clean.drop(['High', 'Low'], axis=1, inplace=True)

data_crypto_clean.reset_index(inplace=True) 
data_electricity_clean.reset_index(inplace=True)
data_crypto_clean.drop('index', axis=1, inplace=True)
data_electricity_clean.drop('index', axis=1, inplace=True)

# Finding the mean of each month from the daily mean of the crypto price
list = [31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,29,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,29,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30]

sum = 0
index = 0
for i in range(len(list)):
    for j in range(list[i]):
        sum += data_crypto_clean['Value'][index]
        index += 1
        if j >= 1:
            data_crypto_clean.drop(index - 1, axis=0, inplace=True)
    data_crypto_clean['Value'][(i * list[i]) + j] = sum / list[i]
    sum = 0

for i in range(len(data_electricity_clean)):
    if str(data_electricity_clean['YYYYMM'][i])[-2:] == '13':
        data_electricity_clean.drop(i,axis=0,inplace=True)

data_crypto_clean.reset_index(inplace=True) 
data_electricity_clean.reset_index(inplace=True)
data_crypto_clean.drop('index', axis=1, inplace=True)
data_electricity_clean.drop('index', axis=1, inplace=True)

#%%
print(len(data_crypto_clean),len(data_electricity_clean))
print(data_crypto_clean)
print(data_electricity_clean)

#%%
# A simple scatter plot with Matplotlib
fig1 = plt.figure(figsize=(10,10))
ax1 = plt.axes()

x_data_crypto = data_crypto_clean[['YYYYMM']]
y_data_crypto = data_crypto_clean[['Value']]

ax1.scatter(x=np.linspace(0,98,98),y=y_data_crypto,alpha=0.5)

ax1.set(xlabel='Date',
       ylabel='Price of Bitcoin (USD)',
       title='Time vs Price of Bitcoin',);
# %%
fig2 = plt.figure(figsize=(10,10))
ax2 = plt.axes()

x_data_electricity = data_electricity_clean[['YYYYMM']]
y_data_electricity = data_electricity_clean[['Value']]

ax2.scatter(np.linspace(0,98,98), data_electricity_clean.Value, color='blue',)

ax2.set(xlabel='Price of Electricity (Cents per Kilowatthour)',
       ylabel='Date',
       title='Time vs Price of Electricity');
# %%
ax1_hist = data_crypto_clean.Value.plot.hist(bins=25)
ax1_hist.set_xlabel('Price of Bitcoin ($USD)')
# %%
ax2_hist = data_electricity_clean.Value.plot.hist(bins=25)
ax2_hist.set_xlabel('Price of Electricity (Cents per Kilowatthour)')
# %%
normaltest(data_crypto_clean.Value)
# %%
normaltest(data_electricity_clean.Value)
#%%
# %%
data_clean = pd.DataFrame()
data_clean['Date'] = data_crypto_clean['YYYYMM']
data_clean['BTC_Price'] = data_crypto_clean['Value']
data_clean['Electricity_Price'] = data_electricity_clean['Value']
data_clean.head()
# %%
fig3 = plt.figure(figsize=(10,10))
ax3 = plt.axes()

ax3.scatter(x=data_clean.Date, y=data_clean.Electricity_Price, alpha=0.5)
ax3.set(xlabel='BTC Price ($USD)',
        ylabel='Electricity Price (Cents per Kilowatthour)',
        title='BTC Price vs Electricity Price')
# %%
