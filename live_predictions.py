"""Rank the universe today with the trained v2 model.

Output score is the predicted percentile (0..1) of each stock's
market-excess return over the next HORIZON_DAYS trading days.
"""
import pandas as pd
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta

from config import STOCKS, BENCHMARK, HORIZON_DAYS, HISTORY_DAYS, MODEL_VERSION
from feature_engineering import FEATURE_COLS, build_panel, cross_sectional_normalize

print("Loading trained model...")
booster = xgb.Booster()
booster.load_model('trained_model.json')

print("Fetching latest data...")
end_date = datetime.now()
# mom_12m_ex1m needs 252 trading days of history, so fetch ~2 years
start_date = end_date - timedelta(days=min(HISTORY_DAYS, 730))

raw = yf.download(STOCKS + [BENCHMARK], start=start_date, end=end_date,
                  progress=False, auto_adjust=True, group_by='ticker')

data = {}
for stock in STOCKS:
    try:
        df = raw[stock].dropna(how='all')
        if len(df) > 300:
            data[stock] = df
    except KeyError:
        pass
spy_data = raw[BENCHMARK].dropna(how='all')
print(f"Usable tickers: {len(data)} / {len(STOCKS)}")

panel = build_panel(data, spy_data, with_target=False)
panel = cross_sectional_normalize(panel)

latest_date = panel['Date'].max()
today = panel[panel['Date'] == latest_date].copy()
print(f"Scoring {len(today)} stocks as of {pd.to_datetime(latest_date).date()}")

today['score'] = booster.predict(xgb.DMatrix(today[FEATURE_COLS]))
results = (today[['stock', 'score']]
           .sort_values('score', ascending=False)
           .reset_index(drop=True))
results['rank'] = range(1, len(results) + 1)

print("=" * 60)
print(f"LIVE RANKINGS - {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"score = predicted percentile of {HORIZON_DAYS}-trading-day market-excess return")
print("=" * 60)
print("\n--- TOP 10 ---\n")
print(results.head(10)[['rank', 'stock', 'score']].to_string(index=False))
print("\n--- BOTTOM 10 ---\n")
print(results.tail(10)[['rank', 'stock', 'score']].to_string(index=False))

filename = f'predictions_v{MODEL_VERSION}_{datetime.now():%Y%m%d}.csv'
results.to_csv(filename, index=False)
print(f"\nSaved to {filename}")
