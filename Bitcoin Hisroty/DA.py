import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:\\Users\\owd1\\OneDrive\\Desktop\\Bitcoin Hisroty\\bitcoin_data_15K.csv')

# Convert 'datetime' to datetime object and set as index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Basic Data Inspection
print(data.info())
print(data.describe())

# Calculate additional technical indicators

# Exponential Moving Average
data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()

# Bollinger Bands
data['BB_upper'] = data['close'].rolling(window=20).mean() + (data['close'].rolling(window=20).std() * 2)
data['BB_lower'] = data['close'].rolling(window=20).mean() - (data['close'].rolling(window=20).std() * 2)

# MACD (Moving Average Convergence Divergence)
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Visualization
plt.figure(figsize=(15, 10))
plt.subplot(3,1,1)
plt.plot(data['close'], label='Close Price')
plt.plot(data['EMA_12'], label='EMA 12')
plt.plot(data['EMA_26'], label='EMA 26')
plt.title('Close Price and Exponential Moving Averages')
plt.legend()

plt.subplot(3,1,2)
plt.plot(data['BB_upper'], label='Bollinger Upper Band')
plt.plot(data['close'], label='Close Price')
plt.plot(data['BB_lower'], label='Bollinger Lower Band')
plt.title('Bollinger Bands')
plt.legend()

plt.subplot(3,1,3)
plt.plot(data['MACD'], label='MACD')
plt.plot(data['MACD_signal'], label='Signal Line')
plt.title('MACD')
plt.legend()

plt.tight_layout()
plt.show()
