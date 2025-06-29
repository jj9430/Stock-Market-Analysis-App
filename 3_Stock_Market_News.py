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


def fetch_stock_news(topic,page_size=5):

    api = "54f96e0f23554593876f2617475c8ada"
    url = "https://newsapi.org/v2/everything"

    params = {

        "q": topic,
        "apiKey": api,
        "pageSize": page_size,
        "sortBy": "relevancy",
        "language": "en"
    }
    response = requests.get(url,params=params)

    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        return articles
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []


    
def main():
    symbolsdf = load_sp500_list()
    symbols = symbolsdf['Symbol'].unique()
    symbol = st.selectbox("Select a Symbol", symbols)
    if symbol:
        articles = fetch_stock_news(symbol)

        for  article in articles:
            st.subheader(article['title'])
            st.write(article['description'])
            st.markdown(f"[Read more]({article['url']})")

if __name__ == "__main__":
    main()


