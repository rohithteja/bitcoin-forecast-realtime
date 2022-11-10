import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error



def LSTM_data(df, lookback):
    df_lstm = pd.DataFrame(df['Close'])
    df_lstm = df_lstm.rename(columns={'Close': 'lag_0'})
    for shift in range(1, lookback):
        df_lstm['lag_{}'.format(shift)] = df_lstm['lag_0'].shift(shift * -1)

    df_lstm = df_lstm.dropna()
    y = df_lstm.iloc[:, -1]
    x = df_lstm.iloc[:, :-1]
    return x, y

def save_metrics(yhat,y_test):
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    # MAE
    mae = mean_absolute_error(y_test, yhat)
    # MAPE
    mape = mean_absolute_percentage_error(y_test, yhat)
    # R2
    r2 = r2_score(y_test, yhat)
    # save metrics
    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
    # to dataframe
    metrics = pd.DataFrame(metrics, index=[0])
    return metrics

def update_metrics():
    # download 5 year crypto prices from Yahoo Finance
    df = yf.download(tickers='BTC-USD', period = '6y', interval = '1d')

    df_train = df[:-90]
    df_test = df[-90:]
    # for artifacts
    X_train, y_train = LSTM_data(df_train, 8)
    n_features = 1
    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], n_features))
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(7, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train.values, epochs=200, verbose=0)

    #  prediction
    X_test, y_test = LSTM_data(df_test, 8)
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], n_features))
    yhat = model.predict(X_test, verbose=0)

    plt.figure(figsize=(15,10),dpi=300)
    plt.plot(y_test.index, y_test.values, label='Original')
    plt.plot(y_test.index, yhat, label='Predicted')
    plt.ylabel('BTC-USD Price')
    plt.legend()
    plt.savefig('artifacts/forecast_90days.png', bbox_inches='tight')

    metrics = save_metrics(yhat,y_test.values)
    metrics.to_csv('artifacts/metrics.csv', index=False)