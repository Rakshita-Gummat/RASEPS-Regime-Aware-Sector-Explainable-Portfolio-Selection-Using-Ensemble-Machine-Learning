import numpy as np

def compute_metrics(portfolio_returns):

    mean_daily = portfolio_returns.mean()
    std_daily = portfolio_returns.std()

    annual_return = (1 + mean_daily)**252 - 1
    sharpe = (mean_daily / std_daily) * np.sqrt(252)

    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Annual Return": annual_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }