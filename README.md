# Stock Ranker ML Model

An XGBoost based machine learning model to rank stocks by predicted 30 day forward returns.

## Overview

This project trains an ML model to rank stocks by predicted future performance.

Training data: 2 years of historical OHLCV data
Target: 30 day forward cumulative return
Model: XGBoost regressor with 24 technical features
Validation: Backtesting and out of sample generalization

## Results

### Original Sector Clean Energy and Growth 50 stocks

Top 10 percent win rate: 74.69%
Bottom 10 percent win rate: 23.84%
Outperformance: 15.85%
Test R squared: 0.0026

### New Sector Tech and Defense 50 stocks

Top 10 percent win rate: 49.65%
Bottom 10 percent win rate: 24.43%
Outperformance: 16.22%
Test R squared: 0.0020

Key insight: Model generalizes across sectors with similar outperformance.

## Live Predictions

Live predictions are made monthly and tracked against actual returns.

### June 2 2026 Predictions Energy Sector

See predictions 20260602 energy.csv

Top pick: PLUG with 5.5 percent predicted return
Bottom pick: MSFT with negative 2.0 percent predicted return

### June 2 2026 Predictions Tech and Defense

See predictions 20260602 tech.csv

Top pick: INTC with 2.05 percent predicted return
Bottom pick: AVGO with negative 2.03 percent predicted return

Note: Actual returns will be validated on July 2 2026 which is 30 days forward.

## Features

The model uses 24 technical indicators including:

Moving averages: SMA 20, SMA 50, SMA 200
Momentum indicators: MACD, RSI, ROC
Volatility measures
Price action features
Volume analysis

See calculate features function in stock ranker.py for full list.

## Files

stock ranker.py: Main training and backtesting pipeline
config.py: Stock lists and hyperparameters
live predictions.py: Generate live predictions
trained model.json: Trained XGBoost model
predictions 20260602 energy.csv: Live predictions for energy sector
predictions 20260602 tech.csv: Live predictions for tech and defense sector

## Validation Timeline

June 2 2026: Initial predictions made and documented in this repository
July 2 2026: Actual 30 day returns collected from market data
July 2 2026: Results added to repository to validate prediction accuracy

## How to Use

Install dependencies:

pip install r requirements.txt

Train model on your own data:

python stock ranker.py

Generate live predictions:

python live predictions.py

## Model Details

The XGBoost model uses the following hyperparameters:
learning rate: 0.01
max depth: 2
min child weight: 20
gamma: 1.0
subsample: 0.6
colsample bytree: 0.6
alpha: 1.0
lambda: 10.0

Early stopping occurs at iteration 6 out of 2000 to prevent overfitting.
Built as a college application project to explore

Machine learning fundamentals
Feature engineering for financial data
Model validation and backtesting
Real world data analysis
Statistical thinking and hypothesis testing

Built by Aakash Bhat in June 2026
