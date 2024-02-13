import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv('C:\\Users\\owd1\\OneDrive\\Desktop\\Bitcoin Hisroty\\bitcoin_data_15K.csv')

# Convert 'datetime' to datetime object and set as index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Feature Engineering
# Calculate percentage changes and log return
data['open_change'] = data['open'].pct_change()
data['close_change'] = data['close'].pct_change()
data['log_return'] = np.log(data['close'] / data['close'].shift(1))
data['relative_diff'] = (data['close'] - data['open']) / data['open']
data['high_low_pct'] = (data['high'] - data['low']) / data['low']

# Drop NaN values created by these transformations
data.dropna(inplace=True)

# Normalize the features
scaler = MinMaxScaler()
features_to_scale = ['open', 'high', 'low', 'volume', 'SMA_12', 'SMA_26', 'RSI', 'open_change', 'close_change', 'log_return', 'relative_diff', 'high_low_pct']
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Define features (X) and target (y)
# Using 'close_change' as the target variable
X = data.drop(['close', 'close_change'], axis=1)
y = data['close_change']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Training
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
