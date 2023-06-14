# Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Loading
df = pd.read_csv('stock_prices.csv')

# Preparation
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

prediction_days = 60
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Training LSTM 
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model_lstm.add(LSTM(units=50, return_sequences=True))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(units=1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(x_train, y_train, epochs=25, batch_size=32)

# Training RF
model_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
model_rf.fit(x_train, y_train)

# Combining LSTM & RF
inputs = df['Close'][len(df) - len(x_train) - prediction_days:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for x in range(prediction_days, len(inputs)):
    x_test.append(inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions_lstm = model_lstm.predict(x_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)

predictions_rf = model_rf.predict(x_test)
predictions_rf = scaler.inverse_transform(predictions_rf.reshape(-1, 1))

predictions_combined = (predictions_lstm + predictions_rf) / 2

# Predicting stock prices
last_day = df['Date'].iloc[-1]
next_30_days = pd.date_range(last_day, periods=31, freq='B')
next_30_days = pd.DataFrame(next_30_days, columns=['Date'])

next_30_days['Close'] = 0
inputs = df['Close'][len(df) - prediction_days:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

for i in range(30):
    x_test = inputs[-prediction_days:]
    x_test = np.reshape(x_test, (1, x_test.shape[0], 1))

    prediction_lstm = model_lstm.predict(x_test)
    prediction_lstm = scaler.inverse_transform(prediction_lstm)

    prediction_rf = model_rf.predict(x_test)
    prediction_rf = scaler.inverse_transform(prediction_rf.reshape(-1, 1))

    prediction_combined = (prediction_lstm + prediction_rf) / 2

    inputs = np.append(inputs, prediction_combined)
    next_30_days['Close'][i] = prediction_combined[0][0]

# Evaluating
rmse_lstm = np.sqrt(np.mean(((predictions_lstm - y_test) ** 2)))
rmse_rf = np.sqrt(np.mean(((predictions_rf - y_test) ** 2)))
rmse_combined = np.sqrt(np.mean(((predictions_combined - y_test) ** 2)))

print('LSTM RMSE:', rmse_lstm)
print('Random Forest RMSE:', rmse_rf)
print('Combined RMSE:', rmse_combined)

# Plotting
plt.plot(df['Date'], df['Close'])
plt.plot(next_30_days['Date'], next_30_days['Close'])
plt.show()
