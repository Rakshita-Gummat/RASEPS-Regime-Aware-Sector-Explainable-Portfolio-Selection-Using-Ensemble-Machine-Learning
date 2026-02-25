from src.data_loader import download_stock_data, load_prices
from src.feature_engineering import create_features, create_target, get_sector_series
from src.regime_detection import detect_market_regime
from src.ensemble import train_ensemble, ensemble_predict
from src.portfolio import probability_weighted_portfolio
from src.evaluation import compute_metrics
from src.explainability import explain_model

# ---------------- TICKERS ----------------
tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA",
           "JPM","BAC","GS","XOM","CVX","JNJ","PFE","WMT","KO"]

# ---------------- LOAD DATA ----------------
download_stock_data(tickers)
prices = load_prices()

# ---------------- FEATURES ----------------
features = create_features(prices)
target = create_target(prices)

common = features.index.intersection(target.index)
X = features.loc[common]
y = target.loc[common]

# ---------------- REGIME ----------------
regime = detect_market_regime(X)

# ---------------- ENSEMBLE TRAIN ----------------
models = train_ensemble(X, y)

# ---------------- PREDICTION ----------------
probs = ensemble_predict(models, X)

# ---------------- PORTFOLIO ----------------
from src.backtest import walk_forward_backtest

portfolio_returns = walk_forward_backtest(
    X, y, prices,
    train_window=500,
    step=20,
    top_pct=0.2
)

print("Average Daily Return:", portfolio_returns.mean())

metrics = compute_metrics(portfolio_returns)

print("\nPerformance Metrics")
for k, v in metrics.items():
    print(k, ":", round(v,4))

#iii
metrics = compute_metrics(portfolio_returns)

print("\nPerformance Metrics")
for k, v in metrics.items():
    print(k, ":", round(v,4))

    # ================= VISUALIZATION SECTION =================

import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# ---------- CUMULATIVE RETURN ----------
cumulative = (1 + portfolio_returns).cumprod()

plt.figure(figsize=(10,5))
plt.plot(cumulative)
plt.title("Cumulative Portfolio Return")
plt.xlabel("Time")
plt.ylabel("Growth")
plt.grid()
plt.savefig("outputs/cumulative_return.png")
plt.close()


# ---------- DRAWDOWN ----------
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak

plt.figure(figsize=(10,5))
plt.plot(drawdown)
plt.title("Portfolio Drawdown")
plt.xlabel("Time")
plt.ylabel("Drawdown")
plt.grid()
plt.savefig("outputs/drawdown.png")
plt.close()


# ---------- MARKET REGIME ----------
plt.figure(figsize=(10,4))
plt.plot(regime)
plt.title("Market Regime Classification")
plt.xlabel("Time")
plt.ylabel("Regime")
plt.grid()
plt.savefig("outputs/regime.png")
plt.close()

print("Visualization plots saved in outputs folder")

# ---------------- EXPLAIN ----------------
explain_model(models[0], X)

print("Pipeline complete")