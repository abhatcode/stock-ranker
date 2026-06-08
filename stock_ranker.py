import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from config import STOCKS, LOOKBACK_DAYS
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Fetch historical data for all stocks
print("Fetching historical data for 50 stocks + SPY benchmark")

end_date = datetime.now()
start_date = end_date - timedelta(days=730) 

data = {}
all_stocks_and_spy = STOCKS + ['SPY']
for stock in all_stocks_and_spy:
    try:
        print(f"  Fetching {stock}...", end=" ")
        df = yf.download(stock, start=start_date, end=end_date, progress=False)
        if len(df) > 0:
            data[stock] = df
            print("✓")
        else:
            print(" (no data)")
    except Exception as e:
        print(f"({str(e)[:20]})")

print(f"\nSuccessfully fetched {len(data)} tickers out of {len(all_stocks_and_spy)}")

spy_data = data.pop('SPY', None)
if spy_data is not None:
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    print("\nCould not fetch SPY data. Relative strength feature will be skipped.")
if len(data) > 0:
    first_stock = list(data.keys())[0]
    print(f"\nSample data for {first_stock}:")
    print(data[first_stock].head(10))
    print(f"\nData shape: {data[first_stock].shape}")
    print(f"Columns: {list(data[first_stock].columns)}")
def calculate_features(df, stock_name, spy_df=None):
    """Calculate all features for one stock's dataframe"""
    
    # Make a copy so we don't modify original
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['daily_return'] = df['Close'].pct_change()
    
    df['momentum_30'] = (df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30)
    df['momentum_60'] = (df['Close'] - df['Close'].shift(60)) / df['Close'].shift(60)
    
    df['volatility_30'] = df['daily_return'].rolling(window=30).std()
    df['volatility_60'] = df['daily_return'].rolling(window=60).std()
    
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    #  Momentum Strength
    df['momentum_strength'] = df['momentum_60'] / df['volatility_30']
    # Price Position in Range
    low_50 = df['Close'].rolling(window=50).min()
    high_50 = df['Close'].rolling(window=50).max()
    df['price_position'] = (df['Close'] - low_50) / (high_50 - low_50)
    #  Rate of Change
    df['roc_acceleration'] = df['momentum_60'] - df['momentum_30']
    #  Volume Confirmation
    df['volume_confirmation'] = df['daily_return'] * df['volume_ratio']
    #  MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    #  Bollinger Band Position
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = df['sma_20'] + (bb_std * 2)
    bb_lower = df['sma_20'] - (bb_std * 2)
    df['bollinger_pos'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    #  Price Acceleration (5-day return)
    df['price_accel_5d'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    #  Volatility Trend
    df['volatility_trend'] = df['volatility_30'] - df['volatility_60']
    # Relative Strength vs Market (SPY)
    if spy_df is not None:
        spy_returns = spy_df['Close'].pct_change()
        df = df.join(spy_returns.rename('spy_return'))
        df['relative_strength'] = df['daily_return'] - df['spy_return']
    # EMA and Normalized Range
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['price_vs_ema_20'] = (df['Close'] - df['ema_20']) / df['ema_20']
    df['range_50'] = (high_50 - low_50) / df['Close']

    #Additional Momentum and Trends
    df['returns_10d'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    df['returns_30d'] = (df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30)
    df['volume_trend_30'] = df['Volume'] / df['Volume'].rolling(window=30).mean()
    df['price_vs_sma_200'] = (df['Close'] - df['sma_200']) / df['sma_200']
    df['sma_20_vs_50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
    df['target'] = (df['Close'].shift(-30) - df['Close']) / df['Close']
    df['Date'] = df.index
    df['stock'] = stock_name
    
    feature_cols = [
        'sma_20', 'sma_50', 'sma_200', 'momentum_60', 'volatility_30', 'volume_ratio', 'rsi',
        'momentum_strength', 'price_position', 'roc_acceleration',
        'volume_confirmation', 'macd', 'bollinger_pos', 'price_accel_5d', 'volatility_trend',
        'relative_strength', 'price_vs_ema_20', 'range_50', 
        'returns_10d', 'returns_30d', 'volume_trend_30', 'price_vs_sma_200', 'sma_20_vs_50',
        'target', 'Date', 'stock'
    ]

    existing_feature_cols = [col for col in feature_cols if col in df.columns]
    df = df[existing_feature_cols].copy()
    # Drop rows with NaN (first 200 rows, last 5 rows)
    df = df.dropna()
    return df
all_features = []
for stock in STOCKS:
    if stock in data:
        print(f"  Processing {stock}...", end=" ")
        features_df = calculate_features(data[stock], stock, spy_data)
        all_features.append(features_df)
        print(f"({len(features_df)} rows)")

df_all = pd.concat(all_features, ignore_index=True)
if 'Date' in df_all.columns:
    df_all = df_all.sort_values('Date').reset_index(drop=True)
elif 'index' in df_all.columns:
    df_all = df_all.rename(columns={'index': 'Date'}).sort_values('Date').reset_index(drop=True)


df_all['target'] = df_all.groupby('Date')['target'].transform(lambda x: x - x.mean())

print(f"\nTotal rows: {len(df_all)}")
print(f"Features shape: {df_all.shape}")
print(df_all.head())
print(df_all.isnull().sum())
print(df_all.describe())

X = df_all.drop(['target', 'stock', 'Date'], axis=1, errors='ignore')
y = df_all['target']
stocks = df_all['stock']

split_index = int(len(X) * 0.8)
X_train_full, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_full, y_test = y.iloc[:split_index], y.iloc[split_index:]
stocks_train_full, stocks_test = stocks.iloc[:split_index], stocks.iloc[split_index:]

val_size = max(1, int(len(X_train_full) * 0.2))
X_train, X_val = X_train_full.iloc[:-val_size], X_train_full.iloc[-val_size:]
y_train, y_val = y_train_full.iloc[:-val_size], y_train_full.iloc[-val_size:]
stocks_train = stocks_train_full.iloc[:-val_size]

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# Create XGBoost model - stronger regularization to reduce overfitting
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 2,
    'min_child_weight': 20,
    'gamma': 1.0,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'alpha': 1.0,
    'lambda': 10.0,
    'seed': 42,
    'tree_method': 'hist',
}

print("\nTraining the model...")
booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=False,
)
print("Model training complete.")
# Save model
import joblib
booster.save_model('trained_model.json')
joblib.dump(booster, 'trained_model.pkl')
print("Model saved to trained_model.json and trained_model.pkl")

print("Making predictions on the test set...")
predictions = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))

print("\n--- Model Performance ---")

train_predictions = booster.predict(dtrain, iteration_range=(0, booster.best_iteration + 1))
val_predictions = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
train_r2 = r2_score(y_train, train_predictions)
val_r2 = r2_score(y_val, val_predictions)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
directional_accuracy = np.mean((predictions > 0) == (y_test > 0))

print(f"Training R-squared (R²): {train_r2:.4f}")
print(f"Validation R-squared (R²): {val_r2:.4f}")
print(f"Test R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.2%}")

print("\n--- Backtesting: Ranking-Based Strategy ---")

dates_test = df_all['Date'][split_index:]

results = pd.DataFrame({
    'date': dates_test.values,
    'stock': stocks_test.values,
    'actual_return': y_test.values,
    'predicted_return': predictions
})

results = results.reset_index(drop=True)
print(f"\nResults dataframe shape: {results.shape}")
print(results.head(10))

stock_results = results.groupby('stock').agg({
    'actual_return': ['mean', 'std', 'count'],
    'predicted_return': 'mean'
}).round(6)

print("\nPer-stock statistics:")
print(stock_results.head(10))

results['predicted_rank'] = results['predicted_return'].rank(pct=True)

top_10_pct = results[results['predicted_rank'] >= 0.9]
bottom_10_pct = results[results['predicted_rank'] < 0.1]
middle_80_pct = results[(results['predicted_rank'] >= 0.1) & (results['predicted_rank'] < 0.9)]

top_10_win_rate = (top_10_pct['actual_return'] > 0).sum() / len(top_10_pct) if len(top_10_pct) > 0 else 0
bottom_10_win_rate = (bottom_10_pct['actual_return'] > 0).sum() / len(bottom_10_pct) if len(bottom_10_pct) > 0 else 0

top_10_avg_return = top_10_pct['actual_return'].mean()
middle_80_avg_return = middle_80_pct['actual_return'].mean()
bottom_10_avg_return = bottom_10_pct['actual_return'].mean()

outperformance = top_10_avg_return - bottom_10_avg_return

print("\n--- Decile Analysis ---")
print(f"Top 10% win rate: {top_10_win_rate:.2%} ({(top_10_pct['actual_return'] > 0).sum()}/{len(top_10_pct)})")
print(f"Bottom 10% win rate: {bottom_10_win_rate:.2%} ({(bottom_10_pct['actual_return'] > 0).sum()}/{len(bottom_10_pct)})")
print(f"Top 10% avg return: {top_10_avg_return:.6f}")
print(f"Middle 80% avg return: {middle_80_avg_return:.6f}")
print(f"Bottom 10% avg return: {bottom_10_avg_return:.6f}")
print(f"Outperformance (Top - Bottom): {outperformance:.6f} ({outperformance*100:.4f}%)")

latest_date = results['date'].max()
print(f"\n--- Current Model Rankings (As of {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}) ---")
latest_predictions = results[results['date'] == latest_date].copy()
latest_predictions = latest_predictions.sort_values(by='predicted_return', ascending=False)
latest_predictions['rank'] = range(1, len(latest_predictions) + 1)

print(latest_predictions[['rank', 'stock', 'predicted_return', 'actual_return']].to_string(index=False))

print("\n--- Feature Importance ---")
feature_scores = booster.get_score(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': [feature_scores.get(feature_name, feature_scores.get(f'f{i}', 0.0)) for i, feature_name in enumerate(X.columns)]
}).sort_values('importance', ascending=False)

print(feature_importance)
print("\n--- Feature Standard Deviations ---")
for col in X_train.columns:
    std_val = X_train[col].std()
    print(f"{col}: {std_val:.6f}")

print("\n--- Target Variance ---")
print(f"Target std in training: {y_train.std():.6f}")
print(f"Target std in test: {y_test.std():.6f}")

print("\n--- Diagnostics ---")
try:
    print(f"Best iteration: {booster.best_iteration}")
    print(f"Stopped at iteration: {booster.best_iteration + 1} out of 2000")
except AttributeError:
    print("Note: best_iteration not available (no early stopping used)")
    print("Model trained with native xgb.train booster")
    
print(f"\nNumber of features: {len(X.columns)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Target (y) range: [{y.min():.6f}, {y.max():.6f}]")
print(f"Target (y) std: {y.std():.6f}")
print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
print(f"Predictions std: {predictions.std():.6f}")

mean_prediction = y_train.mean()
baseline_r2 = r2_score(y_test, [mean_prediction] * len(y_test))
print(f"\nBaseline R² (predicting mean): {baseline_r2:.4f}")
print(f"Model R² vs baseline: {r2 - baseline_r2:.4f}")