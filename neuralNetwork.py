import math

import pandas_datareader as web
import numpy as np
import pandas as pd
import tensorflow.keras.layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# getting the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-12-30')

# print dataframe
# print(df)

# getting number of rows and columns in dataset
# print(df.shape)

# Visualzing the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

# Creating new dataframe with only Close Column
data = df.filter(['Close'])

# Converting Dataframe to numpy array
dataset = data.values

# Getting the number of rows to train the model
training_data_len = math.ceil(len(dataset) * 0.8)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Creating the training data set
# Creating scaled training data set
train_data = scaled_data[0:training_data_len, :]

# Splitting data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    # if i <= 61:
    #     print(x_train)
    #     print(y_train)
    #     print()

# Converting the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building LSTM Model
model = Sequential()
model.add(tensorflow.keras.layers.LSTM(50, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compiling Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Creating the testing data set
# Creating new array containing scaled values
test_data = scaled_data[training_data_len - 60:, :]

# Creating the data_sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Converting data to numpy array
x_test = np.array(x_test)

# Reshaping data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Getting the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)

# Plotting the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize model
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'])
plt.show()

# Getting the quote
stock_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-12-31')

# Creating new Dataframe
new_df = stock_quote.filter(['Close'])

# Getting the last 60 day closing price values and converting the data frame to an array
last_60_days = new_df[-60:].values

# Scale data to be values in between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

# creating empty list
X_test = []

# Appending past 60 days to X_test
X_test.append(last_60_days_scaled)

# Converting X_test data set to numpy array
X_test = np.array(X_test)

# Reshaping data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scaled price
predicted_price = model.predict(X_test)

# undoing the scaling
predicted_price = scaler.inverse_transform(predicted_price)

print(predicted_price)


