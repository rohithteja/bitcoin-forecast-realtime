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
import matplotlib.pyplot as plt

def load_model():
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    return model

df = yf.download(tickers='BTC-USD', period = '6y', interval = '1d')
update_metrics()
retrain_model()
# predict for next 7 days
model = load_model()
pred_list = []
for i in [7,6,5,4,3,2,1]:
    if i == 7:
        X_test = df.tail(i).Close.T.values.reshape((1, 3, 1))
        pred = model.predict(X_test)[0]
        pred_list.append(pred)
    else:
        temp = df.tail(i).Close.T.values
        X_test = np.append(temp, pred_list).reshape((1, 3, 1)).astype(float)

        pred = model.predict(X_test)[0]
        pred_list.append(pred)

# show predictions
st.title('Bitcoin Price Prediction')
st.write('Predictions for next 7 days:')
pred = pd.DataFrame(pd.date_range(df.tail(1).index.values[0], periods=8, freq='D')[1:], columns=['Date'])
pred['Price (USD)'] = pred_list
pred.Date = pd.to_datetime(pred.Date)
pred['Price (USD)'] = pred['Price (USD)'].astype(int)
st.write(pred)

# plot current weeks price and forecast
st.markdown('## Current week price and forecast')
df2 = df.tail(7)
df2['Date'] = df2.index
df2 = df2[['Date', 'Close']]
df2 = df2.rename(columns={'Close': 'Price (USD)'})
df2.Date = pd.to_datetime(df2.Date)

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(df2.Date, df2['Price (USD)'], label='Current week price')
ax.plot(pred.Date, pred['Price (USD)'], label='Forecast')
ax.legend()
ax.set_ylabel('Price (USD)')
st.pyplot(fig)

# model performance
st.markdown('### Model Performance')
df = pd.read_csv('artifacts/metrics.csv')
st.write(df)

st.image('artifacts/forecast_90days.png')


count = st_autorefresh(interval=1440 * 60 * 1000, key="dataframerefresh")

if count == True:
    update_metrics()
    retrain_model()
