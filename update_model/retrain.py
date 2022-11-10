import yfinance as yf
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import pickle



def LSTM_data(df, lookback):
    df_lstm = pd.DataFrame(df['Close'])
    df_lstm = df_lstm.rename(columns={'Close': 'lag_0'})
    for shift in range(1, lookback):
        df_lstm['lag_{}'.format(shift)] = df_lstm['lag_0'].shift(shift * -1)

    df_lstm = df_lstm.dropna()
    y = df_lstm.iloc[:, -1]
    x = df_lstm.iloc[:, :-1]
    return x, y

def retrain_model():
    df = yf.download(tickers='BTC-USD', period = '6y', interval = '1d')
    # retrain model
    X, y = LSTM_data(df, 6)
    n_features = 1
    X = X.values.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(5, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y.values, epochs=200, verbose=0)
    pickle.dump(model, open('artifacts/model.pkl', 'wb'))
