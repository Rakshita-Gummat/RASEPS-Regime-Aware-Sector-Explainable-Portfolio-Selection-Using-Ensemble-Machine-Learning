import yfinance as yf
import pandas as pd
import os

def download_stock_data(tickers, start="2015-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end)

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        else:
            data = data["Close"]

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/stock_prices.csv")
    print("Stock data saved")

    return data

def load_prices():
    return pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)