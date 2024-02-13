#Recurrent Neural Network

#Part 1 -Data Preprocessing
#Importing the libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
 
#Importing the training set
dataset_train = pd.read_csv('C:/Users/Robin Ochieng/Desktop/Data/training_set.csv')
dataset_train['Sales'] = dataset_train['Sales'].str.replace(',', '')
dataset_train['Sales'] = pd.to_numeric(dataset_train['Sales'], errors='coerce')
training_set = dataset_train.iloc[:, 1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a dataset with 60 timesteps and 1 Output
X_train = []
y_train = []
for i in range(60, 2327):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

#Part 2 - Building the RNN
#Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout

#initializing the RNN
regressor = Sequential()

#Adding the first LSTM Layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape =(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding the second LSTM Layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the third LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the fourth LSTM Layer and some dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the Output Layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss='mean_squared_error')

#Fitting the RNN to the Training Set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Part 3 - Making the predictions and visualizing the reselts
#Getting the real stock price of 2017
dataset_test = pd.read_csv("C:/Users/Robin Ochieng/Desktop/Data/test_set.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price of 2017a
dataset_total = pd.concat((dataset_train['Sales'], dataset_test['Sales']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the results 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Prediscted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
