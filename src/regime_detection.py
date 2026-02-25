def detect_market_regime(features):
    vol = features["market_volatility"]
    threshold = vol.rolling(100).mean()
    regime = (vol > threshold).astype(int)
    return regime