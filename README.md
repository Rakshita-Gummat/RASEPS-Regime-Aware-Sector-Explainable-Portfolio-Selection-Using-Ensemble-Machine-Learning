Regime-Aware Sector-Explainable Portfolio Optimization Using AI

An end-to-end machine learning system designed to model non-stationary financial time-series under regime shifts, integrating ensemble learning, volatility-aware classification, probability-weighted allocation, and explainable AI.

1. Problem Statement

Financial markets are inherently non-stationary, with structural shifts driven by volatility regimes, sector-specific trends, and macroeconomic conditions. Traditional predictive models:

Assume static distributions

Ignore regime transitions

Provide limited interpretability

Fail under volatility spikes

This project addresses financial forecasting as a probabilistic classification task under distribution shift, with explicit regime awareness and transparent decision-making.

2. System Overview

The system consists of five core components:

Time-series data engineering

Volatility-based regime detection

Ensemble predictive modeling

Probability-weighted portfolio construction

Explainable AI interpretation

3. Methodology
3.1 Data Engineering

Historical stock data via Yahoo Finance API

Daily return computation

Feature generation:

Momentum indicators

Rolling volatility

Moving averages

Lagged returns

Market-wide signals

Target formulation:

Target = 1 if future return > 0
Target = 0 otherwise

This transforms forecasting into a supervised classification problem.

3.2 Regime Detection

Market regimes are identified using rolling volatility thresholds:

High-volatility regime

Low-volatility regime

This enables adaptation under distribution shifts.

3.3 Predictive Modeling

Models used:

LightGBM — efficient gradient boosting

XGBoost — nonlinear modeling robustness

Logistic Regression — calibrated probabilistic baseline

Each model outputs:

P(Y = 1 | X)

Final ensemble probability:

P_final = w1P1 + w2P2 + w3P3

Weighted averaging reduces variance and improves generalization.

3.4 Portfolio Construction

Rank assets by predicted probability

Select top-confidence assets

Allocate weights proportional to prediction confidence

Rebalance periodically

Evaluation uses walk-forward validation to eliminate look-ahead bias.

3.5 Explainable AI

SHAP values are used to:

Quantify feature contributions

Validate reliance on momentum/volatility

Increase transparency in decision-making

This ensures responsible AI modeling in high-impact environments.

4. Evaluation Framework

To simulate realistic deployment:

Time-aware train/test splits

Rolling walk-forward backtesting

No data leakage

Metrics:

Cumulative Return

Maximum Drawdown

Volatility

Regime-specific performance

Stability across time windows

5. Results

The regime-aware ensemble model demonstrates:

Improved stability across volatility shifts

Reduced prediction variance via ensemble averaging

Enhanced interpretability through SHAP analysis

Better adaptability compared to static allocation strategies

Performance visualizations include:

Cumulative return curve

Drawdown plot

Regime-wise performance comparison

SHAP feature importance

(See /visualizations directory.)

6. Repository Structure
data/                # Raw and processed financial datasets
models/              # Trained models and ensemble logic
backtesting/         # Walk-forward validation framework
visualizations/      # Performance plots and SHAP outputs
notebooks/           # Experimental analysis notebooks
main.py              # End-to-end execution pipeline
requirements.txt     # Dependencies
7. System Architecture

High-level pipeline:

Data → Feature Engineering → Regime Detection → 
Model Training → Ensemble Prediction → 
Portfolio Allocation → Backtesting → Evaluation → Explainability

The system is modular, allowing independent experimentation on each component.

8. Engineering Highlights

Time-series safe validation

Ensemble probability calibration

Modular ML pipeline design

Risk-aware portfolio logic

Explainable AI integration

Clean separation of data, modeling, and evaluation

9. Technical Stack

Python

LightGBM

XGBoost

Scikit-learn

SHAP

Pandas / NumPy

Matplotlib

yfinance

10. Relevance to AI/ML Roles

This project demonstrates:

Handling distribution shift in real-world data

Ensemble learning under uncertainty

Time-series validation discipline

Applied explainable AI

End-to-end ML system engineering

The methodology generalizes beyond finance to any domain involving non-stationary data and probabilistic decision systems.

License

MIT License
