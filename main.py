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

# predict for next 7 days
model = load_model()
df2 = df.copy()
for i in range(91):

    X_test = df2.tail(5).Close.T.values.reshape((1, 5, 1))
    pred = model.predict(X_test)[0]
    date = df2.tail(1).index[0] + pd.DateOffset(days=1)
    df2 = pd.concat([df2, pd.DataFrame(data=pred, index=[date], columns=['Close'])])

# show predictions
st.title('Bitcoin Price Prediction')
st.write('Predictions for next 90 days:')
pred = df2.copy()
pred = pred.tail(90)
pred = pred.reset_index()
pred = pred.rename(columns={'index': 'Date'})
pred = pred[['Date', 'Close']]
pred = pred.rename(columns={'Close': 'Predicted Price (USD)'})
st.write(pred)

# plot current weeks price and forecast
st.markdown('## Current week price and forecast')
df3 = df.tail(90)
df3['Date'] = df3.index
df3 = df3[['Date', 'Close']]
df3 = df3.rename(columns={'Close': 'Predicted Price (USD)'})
df3.Date = pd.to_datetime(df3.Date)

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(df3.Date, df3['Predicted Price (USD)'], label='Current week price')
df4 = pd.concat([df3, pred]).tail(91)
ax.plot(df4.Date, df4['Predicted Price (USD)'], label='Forecast')
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
