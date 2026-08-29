"""Train the cross-sectional stock ranker (v3).

Pipeline:
1. Download ~8 years of daily data for a ~165-stock, 10-sector universe + SPY.
2. Build scale-free + sector-relative features, z-scored per date.
3. Target: percentile rank per date of the 10-trading-day market-excess return.
4. Purged time-based split (gap >= horizon) for honest evaluation:
   train -> early-stop, refit train+val -> score the held-out test period.
5. Retrain on ALL data with the selected round count and save the
   production model used by live_predictions.py.

Config below was selected by a GPU grid + refinement sweep on validation IC
(2026-07-03): universe-rank target beat sector-neutral, depth-2 trees beat
deeper ones, and the rank:pairwise objective and seed ensembling did not help.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import spearmanr

from config import STOCKS, BENCHMARK, HORIZON_DAYS, HISTORY_DAYS, PURGE_DAYS
from feature_engineering import (FEATURE_COLS, build_panel,
                                 cross_sectional_normalize, add_rank_target)

PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.02,
    'max_depth': 2,
    'min_child_weight': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'alpha': 0.5,
    'lambda': 5.0,
    'seed': 42,
    'tree_method': 'hist',
    'device': 'cuda',  # falls back to CPU automatically if no GPU
}
MAX_ROUNDS = 6000
EARLY_STOP = 150

# ---------------------------------------------------------------- fetch data
print(f"Fetching ~{HISTORY_DAYS // 365} years of data for {len(STOCKS)} stocks + {BENCHMARK}...")
end_date = datetime.now()
start_date = end_date - timedelta(days=HISTORY_DAYS)

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

# ------------------------------------------------------------- build dataset
print("Building feature panel...")
panel = build_panel(data, spy_data, with_target=True)
panel = cross_sectional_normalize(panel)
panel = add_rank_target(panel, sector_neutral=False)
print(f"Panel: {len(panel)} rows, {panel['Date'].nunique()} dates, {len(FEATURE_COLS)} features")

# ------------------------------------------------- purged time-based splits
dates = np.array(sorted(panel['Date'].unique()))
n = len(dates)
train_end, val_end = int(n * 0.70), int(n * 0.85)

train_dates = dates[:train_end - PURGE_DAYS]
val_dates = dates[train_end:val_end - PURGE_DAYS]
test_dates = dates[val_end:]
trval_dates = dates[:val_end - PURGE_DAYS]


def subset(date_arr):
    part = panel[panel['Date'].isin(date_arr)]
    return xgb.DMatrix(part[FEATURE_COLS], label=part['target']), part


dtrain, _ = subset(train_dates)
dval, _ = subset(val_dates)
dtest, test_panel = subset(test_dates)
dtrval, _ = subset(trval_dates)

print(f"Train {len(train_dates)} dates ({train_dates[0].date()} .. {train_dates[-1].date()})")
print(f"Val   {len(val_dates)} dates ({val_dates[0].date()} .. {val_dates[-1].date()})")
print(f"Test  {len(test_dates)} dates ({test_dates[0].date()} .. {test_dates[-1].date()})")

# ---------------------------------------------- train w/ early stop, evaluate
print("\nTraining (early stopping on validation)...")
booster = xgb.train(PARAMS, dtrain, num_boost_round=MAX_ROUNDS,
                    evals=[(dval, 'val')], early_stopping_rounds=EARLY_STOP,
                    verbose_eval=False)
best_iter = booster.best_iteration
print(f"Best iteration: {best_iter}")

# Refit on train+val (scaled rounds for the larger set), score held-out test
refit_rounds = int(best_iter * 1.15) + 1
refit = xgb.train(PARAMS, dtrval, num_boost_round=refit_rounds, verbose_eval=False)
test_panel = test_panel.copy()
test_panel['pred'] = refit.predict(dtest)


def daily_metrics(part):
    ics, spreads = [], []
    for _, day in part.groupby('Date'):
        if len(day) < 20:
            continue
        ic, _ = spearmanr(day['pred'], day['fwd_excess_return'])
        ics.append(ic)
        k = max(5, len(day) // 5)
        spreads.append(day.nlargest(k, 'pred')['fwd_excess_return'].mean()
                       - day.nsmallest(k, 'pred')['fwd_excess_return'].mean())
    return np.array(ics), np.array(spreads)


ics, spreads = daily_metrics(test_panel)
print("\n--- Held-out test performance (out-of-sample) ---")
print(f"Mean daily Spearman IC:      {ics.mean():+.4f}")
print(f"IC information ratio:        {ics.mean() / ics.std():+.2f}")
print(f"% days with positive IC:     {(ics > 0).mean():.1%}")
print(f"Mean top-bottom quintile spread over {HORIZON_DAYS} trading days: {spreads.mean():+.3%}")

by_sector = test_panel.groupby('sector')['pred'].mean().sort_values(ascending=False)
print("\nPer-sector mean prediction (flat = unbiased across categories):")
print(by_sector.round(4).to_string())

# ------------------------------------- production model: train on everything
print("\nTraining production model on all data...")
dall = xgb.DMatrix(panel[FEATURE_COLS], label=panel['target'])
prod_rounds = int(best_iter * 1.3) + 1  # scale again: all dates vs 70%
prod = xgb.train(PARAMS, dall, num_boost_round=prod_rounds, verbose_eval=False)
prod.save_model('trained_model.json')
print(f"Production model ({prod_rounds} rounds) saved to trained_model.json")

print("\n--- Feature importance (gain, production model) ---")
scores = prod.get_score(importance_type='gain')
imp = (pd.Series({f: scores.get(f, 0.0) for f in FEATURE_COLS})
       .sort_values(ascending=False))
print(imp.round(2).to_string())
