import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
from update_model.metrics import update_metrics
from update_model.retrain import retrain_model

def load_model():
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    return model

df = yf.download(tickers='BTC-USD', period = '6y', interval = '1d')

# predict for next 5 days
model = load_model()
pred_list = []
for i in [5,4,3,2,1]:
    if i == 5:
        X_test = df.tail(i).Close.T.values.reshape((1, 5, 1))
        pred = model.predict(X_test)[0]
        pred_list.append(pred)
    else:
        temp = df.tail(i).Close.T.values
        X_test = np.append(temp, pred_list).reshape((1, 5, 1)).astype(float)

        pred = model.predict(X_test)[0]
        pred_list.append(pred)

# show predictions
st.title('Bitcoin Price Prediction')
st.write('Predictions for next 5 days:')
pred = pd.DataFrame(pd.date_range(df.tail(1).index.values[0], periods=6, freq='D')[1:], columns=['Date'])
pred['Price (USD)'] = pred_list
pred.Date = pd.to_datetime(pred.Date)
pred.Date = pred.Date.dt.strftime('%Y-%m-%d')
pred['Price (USD)'] = pred['Price (USD)'].astype(int)
st.write(pred)

# plot current weeks price and forecast
st.markdown('## Current week price and forecast')
plt.figure(figsize=(10,5))
plt.plot(df.tail(7).index, df.tail(7).Close, label='Current week price')
plt.plot(pred.Date, pred['Price (USD)'], label='Forecast')
plt.legend()
plt.ylabel('Price (USD)')
st.pyplot()

# model performance
st.write('Model performance:')
df = pd.read_csv('artifacts/metrics.csv')
st.write(df)

st.image('artifacts/forecast_90days.png')


count = st_autorefresh(interval=1440 * 60 * 1000, key="dataframerefresh")

if count == True:
    update_metrics()
    retrain_model()
