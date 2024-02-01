import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import streamlit as st
from datetime import date


# Τίτλος της εφαρμογής Streamlit
st.title('Stock Trend Prediction')

# Δημιουργία πεδίου εισαγωγής για τον χρήστη, προεπιλεγμένο στο 'AAPL'
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Χρήση του date_input για να επιλέγει ο χρήστης την ημερομηνία έναρξης και λήξης
start = st.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
end = st.date_input('End Date', value=pd.to_datetime('2023-12-31'), max_value=pd.to_datetime('2023-12-31'))
# Ανάκτηση δεδομένων της μετοχής από το Yahoo Finance
df = yf.download(user_input, start=start, end=end)

# Υπολογισμός των μέσων τιμών κινήσεων για 100 και 200 ημέρες
ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()


# Προβολή στατιστικών των δεδομένων στο Streamlit
st.subheader('Data from 2010 - present')
st.write(df.describe())

# Προβολή του γραφήματος τιμών κλεισίματος
st.subheader('Closing Price vs Time chart')
fig1 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.title('Closing Price over Time')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig1)

# Προβολή του γραφήματος τιμών κλεισίματος με 100MA
st.subheader('Closing Price vs Time chart with 100MA')
fig2 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100-day MA', color='orange')
plt.title('Closing Price with 100-day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Προβολή του γραφήματος τιμών κλεισίματος με MA100 και MA200
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Close'], label='Closing Price')
ax.plot(ma100, label='100-day MA')
ax.plot(ma200, label='200-day MA')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)