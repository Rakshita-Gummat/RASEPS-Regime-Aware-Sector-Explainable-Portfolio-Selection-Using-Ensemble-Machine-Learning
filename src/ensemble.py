import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

def train_ensemble(X, y):

    lgb_model = lgb.LGBMClassifier(n_estimators=300)
    xgb_model = xgb.XGBClassifier(eval_metric="logloss")
    log_model = LogisticRegression(max_iter=200)

    lgb_model.fit(X, y)
    xgb_model.fit(X, y)
    log_model.fit(X, y)

    return lgb_model, xgb_model, log_model

def ensemble_predict(models, X):
    lgb_m, xgb_m, log_m = models

    p1 = lgb_m.predict_proba(X)[:,1]
    p2 = xgb_m.predict_proba(X)[:,1]
    p3 = log_m.predict_proba(X)[:,1]

    return 0.5*p1 + 0.3*p2 + 0.2*p3