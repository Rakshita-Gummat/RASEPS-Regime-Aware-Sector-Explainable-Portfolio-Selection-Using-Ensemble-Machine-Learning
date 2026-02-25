import numpy as np
import pandas as pd
from src.ensemble import train_ensemble, ensemble_predict
from src.portfolio import probability_weighted_portfolio

def walk_forward_backtest(features, target, prices,
                          train_window=500,
                          step=20,
                          top_pct=0.2):

    all_returns = []

    for start in range(train_window, len(features) - step, step):

        # ----- TRAIN WINDOW -----
        X_train = features.iloc[start-train_window:start]
        y_train = target.iloc[start-train_window:start]

        # ----- TEST WINDOW -----
        X_test = features.iloc[start:start+step]
        price_test = prices.iloc[start:start+step]

        # ----- TRAIN MODEL -----
        models = train_ensemble(X_train, y_train)

        # ----- PREDICT -----
        probs = ensemble_predict(models, X_test)

        # ----- PORTFOLIO -----
        port_returns, _ = probability_weighted_portfolio(
            probs, price_test, top_pct=top_pct
        )

        all_returns.append(port_returns)

    return pd.concat(all_returns)