import numpy as np
import pandas as pd

def probability_weighted_portfolio(probabilities, price_df, top_pct=0.2):
    
    # convert to numpy
    probs = np.array(probabilities)

    # select top percentage
    cutoff = int(len(probs) * top_pct)
    top_idx = np.argsort(probs)[-cutoff:]

    # create weights (only top assets)
    weights = np.zeros_like(probs)
    weights[top_idx] = probs[top_idx]

    # normalize
    weights = weights / weights.sum()

    # compute returns
    asset_returns = price_df.pct_change().dropna()

    # align length
    weights = weights[:asset_returns.shape[1]]

    portfolio_returns = asset_returns.dot(weights)

    return portfolio_returns, weights