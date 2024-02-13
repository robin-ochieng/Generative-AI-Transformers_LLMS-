import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and preprocess the data
sales_data = pd.read_csv('C:/Users/Robin Ochieng/Desktop/Data/New Business 2022 - 2023.csv',  index_col='Date', parse_dates=True)
sales_data['Sales'] = sales_data['Sales'].str.replace(',', '')
sales_data['Sales'] = pd.to_numeric(sales_data['Sales'], errors='coerce')
sales_data = sales_data.resample('D').sum()
sales_data = sales_data[sales_data != 0].dropna(subset=['Sales'])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(sales_data['Sales'].values.reshape(-1, 1))
scaled_data = pd.DataFrame(scaled_values, index=sales_data.index, columns=['Sales'])

# Create the dataset with look back
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 14
X, Y = create_dataset(scaled_data.values, look_back)
print(X)
# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
print('Train Score: %.2f RMSE' % (train_score))
test_score = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
print('Test Score: %.2f RMSE' % (test_score))

# Calculate mean absolute error
mae_train_score = mean_absolute_error(Y_train[0], train_predict[:,0])
print('Train Score: %.2f MAE' % (mae_train_score))
mae_test_score = mean_absolute_error(Y_test[0], test_predict[:,0])
print('Test Score: %.2f MAE' % (mae_test_score))

# Number of days to forecast
forecast_days = (pd.to_datetime('2024-06-30') - pd.to_datetime('2024-01-01')).days

# Use the last `look_back` days from the training set to make the first prediction
input_data = X_test[-look_back:]

# Reshape the input data to match the model's input shape
input_data = np.reshape(input_data, (1, 1, look_back))

# Initialize an empty list to store the forecasts
forecasts = []

# Forecast the next `forecast_days` days
for _ in range(forecast_days):
    # Make a prediction with the model
    prediction = model.predict(input_data)
    
    # Append the prediction to the forecasts
    forecasts.append(prediction[0, 0])
    
    # Update the input data to include the new prediction, and drop the oldest prediction
    input_data = np.append(input_data[:, :, 1:], prediction.reshape((1, 1, 1)), axis=2)

# Rescale the forecasts back to the original scale
forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

# Get the last date in the original data
last_date = sales_data.index[-1]

# Generate the dates for the forecasts
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=len(forecasts))

# Create a DataFrame with the forecasts
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Sales': forecasts.flatten()
})

print(forecast_df)


print(len(forecast_dates))
print(len(forecasts))