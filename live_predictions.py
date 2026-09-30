"""Scores the universe today with a trained model.

--arch single (default) uses trained_model.json from stock_ranker.py.
--arch moe uses the gated sector experts in models/ from moe_ranker.py.
Both can be run on the same day; each writes its own predictions_v*.csv.
"""
import argparse

import pandas as pd
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta

from config import (STOCKS, BENCHMARK, HISTORY_DAYS, MODEL_VERSION,
                    MACRO_TICKERS, TARGET_MODE, MODEL_ARCH)
from feature_engineering import FEATURE_COLS, build_panel, cross_sectional_normalize
import moe

parser = argparse.ArgumentParser()
parser.add_argument('--arch', choices=['single', 'moe'], default=MODEL_ARCH)
args = parser.parse_args()

print(f"Loading trained model ({args.arch})...")
if args.arch == 'moe':
    experts, supervisor, conductor = moe.load_moe()
    version = 6
else:
    booster = xgb.Booster()
    booster.load_model('trained_model.json')
    version = MODEL_VERSION

print("Fetching latest data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=min(HISTORY_DAYS, 730))

raw = yf.download(STOCKS + [BENCHMARK] + MACRO_TICKERS, start=start_date, end=end_date,
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
macro = {t: raw[t].dropna(how='all') for t in MACRO_TICKERS}
print(f"Usable tickers: {len(data)} / {len(STOCKS)}")

panel = build_panel(data, spy_data, with_target=False, macro=macro)
panel = cross_sectional_normalize(panel)

latest_date = panel['Date'].max()
today = panel[panel['Date'] == latest_date].copy()
print(f"Scoring {len(today)} stocks as of {pd.to_datetime(latest_date).date()}")

if args.arch == 'moe':
    today = today.sort_values('sector', kind='stable')
    today['pred'] = moe.predict_experts(experts, today)
    sup_today = moe.supervisor_frame(today, spy_data, macro)
    sup_today['sup'] = supervisor.predict(xgb.DMatrix(sup_today[moe.SUPERVISOR_COLS]))
    conf = None
    if conductor['confidence']:
        hist = build_panel(data, spy_data, with_target=True, macro=macro)
        hist = cross_sectional_normalize(hist)
        hist = hist[hist['Date'] >= sorted(hist['Date'].unique())[-100]]
        hist = hist.sort_values(['Date', 'sector'], kind='stable')
        monitors, _, _ = moe.load_moe(prefix='monitor_')
        hist['pred'] = moe.predict_experts(monitors, hist)
        by_sector = moe.current_confidence(hist, 'pred')
        conf = today['sector'].map(by_sector).fillna(1.0)
        dialled = sorted(by_sector[by_sector < 1].index)
        print(f"Experts dialled down (negative recent IC): {dialled or 'none'}")
    today['score'] = moe.combine(today, 'pred', sup_today, 'sup',
                                 conductor['lambda'], conf)
    print(sup_today.set_index('sector')['sup'].pipe(lambda x: (x - x.mean()) / x.std())
          .sort_values(ascending=False).round(2).to_string())
    today['score'] = today['score'].rank(pct=True)
else:
    today['score'] = booster.predict(xgb.DMatrix(today[FEATURE_COLS]))
    if TARGET_MODE == 'pairwise_sector':
        today['score'] = today['score'].rank(pct=True)

results = (today[['stock', 'sector', 'score']]
           .sort_values('score', ascending=False)
           .reset_index(drop=True))
results['rank'] = range(1, len(results) + 1)
results['sector_rank'] = (results.groupby('sector')['score']
                          .rank(ascending=False, method='first').astype(int))

cols = ['rank', 'stock', 'sector', 'sector_rank', 'score']
print(results.head(10)[cols].to_string(index=False))
print(results.tail(10)[cols].to_string(index=False))
top3 = results[results['sector_rank'] <= 3].sort_values(['sector', 'sector_rank'])
print(top3[['sector', 'sector_rank', 'stock', 'rank', 'score']].to_string(index=False))

filename = f'predictions_v{version}_{datetime.now():%Y%m%d}.csv'
results.to_csv(filename, index=False)
print(f"Saved to {filename}")
