import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
@st.cache_data
def load_sp500_list():
    df = pd.read_csv('sp500_companies.csv')
    tickers = df['Symbol'].dropna().astype(str).str.strip().tolist()

    fulldata = pd.DataFrame()

    for tick in tickers:
        try:
            t = yf.Ticker(tick)
            data = t.history(period="1y")
            if not data.empty:
                data['Symbol'] = tick 
                fulldata = pd.concat([fulldata, data])
        except Exception as e:
            print(f'Error with {tick}: {e}')

    return fulldata

def fetch_stock_data(symbol, period="1y"):
    sp_500 = load_sp500_list()
    dataframe = sp_500['Symbol'].values
    assert symbol in dataframe
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)

    return data


def calculate_technical_indicators(fulldata):
    fulldata['SMA_20'] = fulldata['Close'].rolling(window=20).mean()
    fulldata['SMA_50'] = fulldata['Close'].rolling(window=50).mean()

    delta = fulldata['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    fulldata['RSI'] = rsi

    fulldata['BB_Middle'] = fulldata['Close'].rolling(window=20).mean()
    std = fulldata['Close'].rolling(window=20).std()
    fulldata['BB_Upper'] = fulldata['BB_Middle'] + 2 * std
    fulldata['BB_Lower'] = fulldata['BB_Middle'] - 2 * std

    fulldata = fulldata.dropna()
    fulldata = fulldata.drop(['Dividends','Symbol', 'Stock Splits'],axis=1)
    st.write(fulldata)
    return fulldata


def ml_model(df):

    X = df[['Open','High','Low','SMA_20','SMA_50','RSI','BB_Lower','BB_Upper','BB_Middle']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

    lm = LinearRegression()

    lm.fit(X_train, y_train)

    prediction = lm.predict(X_test)

    st.write("R² Score:", r2_score(y_test, prediction))
    st.write("MSE:", mean_squared_error(y_test, prediction))

    fig, ax = plt.subplots()

    ax.scatter(y_test, prediction, alpha=0.5)
    ax.set_xlabel("Actual Close Price")
    ax.set_ylabel("Predicted Close Price")
    ax.set_title("Linear Regression: Actual vs Predicted")
    st.pyplot(fig)


def main():

    df = load_sp500_list()
    data = calculate_technical_indicators(df)
    ml_model(data)

    




if __name__ == "__main__":
    main()
