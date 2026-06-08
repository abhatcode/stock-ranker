import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timedelta
from config import STOCKS, LOOKBACK_DAYS
from feature_engineering import calculate_features

try:
    # Load saved model
    print("Loading trained model...")
    booster = xgb.Booster()
    booster.load_model('trained_model.json')
    print("✓ Model loaded\n")

    # Fetch TODAY'S data for all stocks
    print("Fetching latest data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    data = {}
    for stock in STOCKS + ['SPY']:
        try:
            df = yf.download(stock, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                data[stock] = df
                print(f"  ✓ {stock}")
        except Exception as e:
            print(f"  ✗ {stock}: {str(e)[:30]}")

    print(f"\n✓ Fetched {len(data)} stocks\n")

    spy_data = data.pop('SPY', None)

    # Calculate features for each stock
    print("Calculating features...")
    today_features = []
    for stock in data.keys():
        try:
            df = data[stock]
            features_df = calculate_features(df, stock, spy_data)
            if len(features_df) > 0:
                today_row = features_df.iloc[-1:].copy()  # Get last row (today)
                today_features.append(today_row)
        except Exception as e:
            print(f"  Error with {stock}: {str(e)[:50]}")

    print(f"✓ Calculated features for {len(today_features)} stocks\n")

    # Combine all today's features
    all_today = pd.concat(today_features, ignore_index=True)
    print(f"✓ Combined data shape: {all_today.shape}\n")

    # Prepare features for prediction
    stock_names = all_today['stock'].values
    X_today = all_today.drop(['target', 'stock', 'Date'], axis=1, errors='ignore')
    if 'relative_strength' not in X_today.columns:
        X_today['relative_strength'] = 0.0
    print("  Note: relative_strength set to 0 (SPY data not available)")
    print(f"✓ Features shape: {X_today.shape}\n")

    # Create DMatrix for XGBoost (no scaling needed)
    print("Creating prediction matrix...")
    dmatrix_today = xgb.DMatrix(X_today)
    print("✓ Matrix created\n")

    # Make predictions
    print("Making predictions...")
    predictions = booster.predict(dmatrix_today)
    print(f"✓ Predictions made\n")

    # Create results dataframe and rank
    results = pd.DataFrame({
        'stock': stock_names,
        'predicted_return': predictions
    })
    results = results.sort_values('predicted_return', ascending=False).reset_index(drop=True)
    results['rank'] = range(1, len(results) + 1)

    # Print results
    print("=" * 60)
    print(f"LIVE PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n--- TOP 10 STOCKS TO BUY ---\n")
    print(results.head(10)[['rank', 'stock', 'predicted_return']].to_string(index=False))
    
    print("\n--- BOTTOM 10 STOCKS TO AVOID ---\n")
    print(results.tail(10)[['rank', 'stock', 'predicted_return']].to_string(index=False))

    # Save predictions
    filename = f'predictions_{datetime.now().strftime("%Y%m%d")}.csv'
    results.to_csv(filename, index=False)
    print(f"\n✓ Saved to {filename}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()