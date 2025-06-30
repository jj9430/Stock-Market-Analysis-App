import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
import requests
import json
import datetime as dt
import mplfinance as mpf
from scipy.stats import zscore
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
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

    return fulldata

def calculate_volatility_metrics(fulldata):

    returns = fulldata['Close'].pct_change(fill_method=None).dropna()
    
    metrics = {
        "daily_volatility" : returns.std(),
        "annualized_volatility" : returns.std() * np.sqrt(252),
        "var_95" : np.percentile(returns, 5), 
        "var_99" : np.percentile(returns, 1),  
        "max_drawdown" : (((1 + returns).cumprod())/(((1 + returns).cumprod()).cummax() - 1)).min(),
        "sharpe_ratio" : ((returns - (0.01 / 252)).mean() / (returns - (0.01 / 252)).std()) * np.sqrt(252),
            }
    
    return metrics

def detect_anomalies(fulldata):
    numeric_data = fulldata.select_dtypes(include=[np.number])
    
    clean_data = numeric_data.dropna()

    zscores = pd.DataFrame(
        zscore(clean_data),
        columns=clean_data.columns,
        index=clean_data.index
    )

    return zscores
def is_valid_symbol(symbol):
    data = load_sp500_list()
    symbols = data['Symbol'].unique()
    if symbol in symbols:
        return True
    return False


def formatted_dataframe_for_insights(df, symbol, num_rows=5):
    formatted_string = "Market Data:\n\n"


    symbol_data = df[df['Symbol'] == symbol]

    if symbol_data.empty:
        formatted_string += f"--- No data found for symbol: {symbol} ---\n\n"
        return 

    sample_df = symbol_data.sort_index(ascending=False).head(num_rows)
    formatted_string += f"--- Data for {symbol} (Latest {len(sample_df)} days) ---\n"

    for index, row in sample_df.iterrows():
        RSI = f"{row.get('RSI', np.nan):.2f}" if "RSI" in row and pd.notna(row.get('RSI')) else "N/A"
        SMA_20 = f"{row.get('SMA_20', np.nan):.2f}" if "SMA_20" in row and pd.notna(row.get('SMA_20')) else "N/A"
        SMA_50 = f"{row.get('SMA_50', np.nan):.2f}" if "SMA_50" in row and pd.notna(row.get('SMA_50')) else "N/A"
        BB_Upper = f"{row.get('BB_Upper', np.nan):.2f}" if "BB_Upper" in row and pd.notna(row.get('BB_Upper')) else "N/A"
        BB_Middle = f"{row.get('BB_Middle', np.nan):.2f}" if "BB_Middle" in row and pd.notna(row.get('BB_Middle')) else "N/A"
        BB_Lower = f"{row.get('BB_Lower', np.nan):.2f}" if "BB_Lower" in row and pd.notna(row.get('BB_Lower')) else "N/A"

        formatted_string += f"  RSI = {RSI}\n"
        formatted_string += f"  SMA_20 = {SMA_20}\n" 
        formatted_string += f"  SMA_50 = {SMA_50}\n"
        formatted_string += f"  BB_Upper = {BB_Upper}\n"
        formatted_string += f"  BB_Middle = {BB_Middle}\n"
        formatted_string += f"  BB_Lower = {BB_Lower}\n" 
        formatted_string += "\n" 

    formatted_string += "Please generate a market commentary based on the provided data."

    return formatted_string    



def generate_trade_insights(symbol):
    if symbol:
        stock_data = load_sp500_list()
        indicators = calculate_technical_indicators(stock_data)
        llm_text = formatted_dataframe_for_insights(indicators, symbol)

        client = genai.Client(api_key="AIzaSyDZV4_7rW6JiKt994PtSpGSIGmdqzNrK1I")
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            max_output_tokens=500,
            system_instruction="You a stock market expert who can analyze stock indicators and give trade insights."),
        contents=llm_text
        )
        st.write(response.text)
        
def disable_button(label, disabled=True, key=None):
    return st.button(label, disabled=disabled, key=key)


def main():
    st.title("Trade Insights")
    symbolsdf = load_sp500_list()
    symbols = symbolsdf['Symbol'].unique()
    symbol = st.selectbox("Select a Symbol", symbols)
    if symbol:        
        valid = is_valid_symbol(symbol)
        if valid:
                generate_trade_insights(symbol)
        else:
            st.error(f"Invalid Symbol")


if __name__ == "__main__":
    main()
