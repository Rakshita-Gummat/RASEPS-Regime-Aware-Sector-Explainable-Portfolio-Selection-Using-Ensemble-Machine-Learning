import lightgbm as lgb

def train_sector_models(X, y, sector_series):
    models = {}

    for sector in sector_series.unique():
        tickers = sector_series[sector_series == sector].index
        cols = [c for c in X.columns]

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=50,
            random_state=42
        )

        model.fit(X[cols], y)
        models[sector] = model

    return models