import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import datetime

# Set the title of the Streamlit app
st.title('Stock Trend Prediction')

# Create an input field for the user, default to 'AAPL'
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Define the period for data retrieval
start = '2010-01-01'
end = datetime.today().strftime('%Y-%m-%d')

# Retrieve stock data from Yahoo Finance
df = yf.download(user_input, start=start, end=end)

# Display data statistics in Streamlit
st.subheader('Data from 2010 - present')
st.write(df.describe())

ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()

# Plot the closing price vs time chart
st.subheader('Closing Price vs Time chart')
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(df['Close'], label='Closing Price')
ax1.set_title('Closing Price over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)

# Plot the closing price vs time chart with 100MA
st.subheader('Closing Price vs Time chart with 100MA')
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(df['Close'], label='Closing Price')
ax2.plot(ma100, label='100-day MA', color='orange')
ax2.set_title('Closing Price with 100-day Moving Average')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)

# Plot the closing price vs time chart with 100MA & 200MA
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(df['Close'], label='Closing Price')
ax3.plot(ma100, label='100-day MA')
ax3.plot(ma200, label='200-day MA')
ax3.set_title('Closing Price with 100-day & 200-day Moving Averages')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)

data_training = pd.DataFrame (df [ 'Close'][0:int(len (df)*0.70)])
data_testing= pd.DataFrame(df [ 'Close' ][int(len (df) *0.70): int(len (df))])

scaler = MinMaxScaler (feature_range=(0,1))


#splitting the data into x_train and y_train
data_training_array = scaler.fit_transform(data_training)

# Load the trained model
model = load_model('keras_model.h5')



past_100_days = data_training.tail(100)
final_df = pd.concat([data_training, data_testing], ignore_index=True)
input_data = scaler. fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape [0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict (x_test)
scaler= scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot the original price vs predicted price

st.subheader('Predicted vs Actual Price')
fig4, ax4 = plt.subplots(figsize=(12,6))
ax4.plot(y_test, 'b', label='Original Price')
ax4.plot(y_predicted, 'r', label='Predicted Price')
ax4.set_title('Predicted vs Actual Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)
