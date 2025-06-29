from polygon import RESTClient
import yfinance as yf
import pandas as pd
import numpy as np
import json
from typing import cast
from urllib3 import HTTPResponse
from plotly import graph_objects as go
import talib
import streamlit as st
@st.cache_data
def load_sp500_list():
    df = pd.read_csv('/Users/joshuajoseph/Downloads/archive/sp500_companies.csv')
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

def is_valid_symbol(symbol):
    data = load_sp500_list()
    symbols = data['Symbol'].unique()
    if symbol in symbols:
        return True
    return False

@st.cache_data
def create_graph(symbol):
    if symbol:
        valid = is_valid_symbol(symbol)
        if valid:
            client = RESTClient('odriBZODb0_ihmSUo4Va0qnT87ltpYpe')

            aggs = cast(
                HTTPResponse,
                client.get_aggs(
                    symbol,
                    1,
                    'day',
                    '2025-01-01',
                    '2025-06-11',
                    raw=True
                )
            )
            

            json_string = aggs.data.decode('utf-8')
            data = json.loads(json_string)

            
            for item in data:
                if item == 'results':
                    rawdata = data[item]

            closeList = []
            openList = []
            highList = []
            lowList = []
            timesList = []
            for bar in rawdata:
                for category in bar:
                    if category == 'c':
                        closeList.append(bar[category])
                    elif category == 'h':
                        highList.append(bar[category])
                    elif category == 'l':
                        lowList.append(bar[category])
                    elif category == 'o':
                        openList.append(bar[category])
                    elif category == 't':
                        timesList.append(bar[category])

            closeList = np.array(closeList)
            upper, middle,lower = talib.BBANDS(closeList,timeperiod=20,nbdevdn=2,matype=0)

            times = []
            for time in timesList:
                times.append(pd.Timestamp(time, tz='America/Los_Angeles', unit='ms'))
            st.write("Graph for "+symbol)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=times,open=openList,high=highList,low=lowList,close=closeList, name='Market Data'))
            fig.add_trace(go.Scatter(x=times,y=upper, line=dict(color='blue'), name = 'BB Upper'))
            fig.add_trace(go.Scatter(x=times,y=middle, line=dict(color='green'), name = 'BB Upper'))
            fig.add_trace(go.Scatter(x=times,y=lower, line=dict(color='red'), name = 'BB Upper'))

            fig.update_layout(xaxis_rangeslider_visible=False)

            return fig
        else:
            st.error("Invalid Symbol")
            return


  
def main():
    symbolsdf = load_sp500_list()
    symbols = symbolsdf['Symbol'].unique()
    symbol = st.selectbox("Select a Symbol", symbols)
    if symbol:
        valid = is_valid_symbol(symbol)
        if valid:        
            fig = create_graph(symbol)
        
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Symbol does not work")
        else:
            st.error(f"Could not generate chart for {symbol}. Check console for details.")
    else:
        return

if __name__ == "__main__":
    main()


