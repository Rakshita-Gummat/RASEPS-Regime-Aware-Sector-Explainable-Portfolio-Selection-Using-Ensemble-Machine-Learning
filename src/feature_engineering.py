import pandas as pd

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","AMZN":"Technology",
    "NVDA":"Technology","META":"Technology","TSLA":"Automotive",
    "JPM":"Finance","BAC":"Finance","GS":"Finance",
    "XOM":"Energy","CVX":"Energy",
    "JNJ":"Healthcare","PFE":"Healthcare",
    "WMT":"Consumer","KO":"Consumer"
}

def create_features(prices):
    returns = prices.pct_change()
    features = pd.DataFrame(index=prices.index)

    features["momentum_5"] = returns.rolling(5).mean().mean(axis=1)
    features["momentum_20"] = returns.rolling(20).mean().mean(axis=1)
    features["volatility_20"] = returns.rolling(20).std().mean(axis=1)

    ma10 = prices.rolling(10).mean().mean(axis=1)
    ma50 = prices.rolling(50).mean().mean(axis=1)
    features["trend"] = ma10 - ma50

    features["market_return"] = returns.mean(axis=1)
    features["market_volatility"] = returns.std(axis=1)

    # lag memory
    features["lag_1"] = returns.shift(1).mean(axis=1)
    features["lag_5"] = returns.shift(5).mean(axis=1)

    return features.dropna()

def create_target(prices):
    future = prices.pct_change(20).shift(-20).mean(axis=1)
    return (future > 0).astype(int).dropna()

def get_sector_series(prices):
    return pd.Series({t: SECTOR_MAP[t] for t in prices.columns})