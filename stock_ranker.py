"""Trains the single-model stock ranker (v5) and saves trained_model.json.

Downloads price history, builds features, fits an XGBoost ranker on a
purged train/val/test split, and reports honest out-of-sample performance.
Use --target to compare target modes, e.g.:
    python stock_ranker.py --target universe --data-cache raw.pkl
    python stock_ranker.py --target pairwise_sector --data-cache raw.pkl
"""
import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta

from config import (STOCKS, BENCHMARK, HORIZON_DAYS, HISTORY_DAYS, PURGE_DAYS,
                    MACRO_TICKERS, INTERACTION_MODE, TARGET_MODE)
from feature_engineering import (FEATURE_COLS, build_panel,
                                 cross_sectional_normalize, add_rank_target,
                                 interaction_constraints)
from metrics import daily_ic, sector_ic, daily_spread

TARGET_MODES = {
    'universe': ('universe', False),
    'sector_rank': ('sector', False),
    'blend': ('blend', False),
    'pairwise_sector': ('sector', True),
}

parser = argparse.ArgumentParser()
parser.add_argument('--interactions', choices=['none', 'sector'],
                    default=INTERACTION_MODE)
parser.add_argument('--target', choices=list(TARGET_MODES), default=TARGET_MODE)
parser.add_argument('--data-cache',
                    help='pickle of the raw download; reused if it exists')
args = parser.parse_args()
save_production = (args.interactions == INTERACTION_MODE
                   and args.target == TARGET_MODE)
target_kind, use_groups = TARGET_MODES[args.target]

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
    'device': 'cuda',
}
if use_groups:
    PARAMS['objective'] = 'rank:pairwise'
    PARAMS['eval_metric'] = 'auc'
constraints = interaction_constraints(args.interactions)
if constraints:
    PARAMS['interaction_constraints'] = constraints
print(f"Interaction mode: {args.interactions}   Target mode: {args.target}")

MAX_ROUNDS = 6000
EARLY_STOP = 150

print(f"Fetching ~{HISTORY_DAYS // 365} years of data for {len(STOCKS)} stocks + {BENCHMARK}...")
end_date = datetime.now()
start_date = end_date - timedelta(days=HISTORY_DAYS)

if args.data_cache and os.path.exists(args.data_cache):
    raw = pd.read_pickle(args.data_cache)
    print(f"Loaded cached download from {args.data_cache}")
else:
    raw = yf.download(STOCKS + [BENCHMARK] + MACRO_TICKERS, start=start_date, end=end_date,
                      progress=False, auto_adjust=True, group_by='ticker')
    if args.data_cache:
        raw.to_pickle(args.data_cache)

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

panel = build_panel(data, spy_data, with_target=True, macro=macro)
panel = cross_sectional_normalize(panel)
panel = add_rank_target(panel, mode=target_kind)
panel = panel.sort_values(['Date', 'sector'], kind='stable').reset_index(drop=True)
print(f"Panel: {len(panel)} rows, {panel['Date'].nunique()} dates, {len(FEATURE_COLS)} features")

dates = np.array(sorted(panel['Date'].unique()))
n = len(dates)
train_end, val_end = int(n * 0.70), int(n * 0.85)

train_dates = dates[:train_end - PURGE_DAYS]
val_dates = dates[train_end:val_end - PURGE_DAYS]
test_dates = dates[val_end:]
trval_dates = dates[:val_end - PURGE_DAYS]


def make_dmatrix(part):
    if not use_groups:
        return xgb.DMatrix(part[FEATURE_COLS], label=part['target'])
    qid = part.groupby(['Date', 'sector'], sort=True).ngroup().to_numpy()
    return xgb.DMatrix(part[FEATURE_COLS], label=part['target'], qid=qid)


def subset(date_arr):
    part = panel[panel['Date'].isin(date_arr)]
    return make_dmatrix(part), part.copy()


dtrain, _ = subset(train_dates)
dval, val_panel = subset(val_dates)
dtest, test_panel = subset(test_dates)
dtrval, _ = subset(trval_dates)

print(f"Train {len(train_dates)} dates ({train_dates[0].date()} .. {train_dates[-1].date()})")
print(f"Val   {len(val_dates)} dates ({val_dates[0].date()} .. {val_dates[-1].date()})")
print(f"Test  {len(test_dates)} dates ({test_dates[0].date()} .. {test_dates[-1].date()})")

print("Training (early stopping on validation)...")
booster = xgb.train(PARAMS, dtrain, num_boost_round=MAX_ROUNDS,
                    evals=[(dval, 'val')], early_stopping_rounds=EARLY_STOP,
                    verbose_eval=False)
best_iter = booster.best_iteration
print(f"Best iteration: {best_iter}")

val_panel['pred'] = booster.predict(dval, iteration_range=(0, best_iter + 1))
print(f"Validation mean daily IC:    {daily_ic(val_panel).mean():+.4f}")
print(f"Validation within-sector IC: {sector_ic(val_panel).mean():+.4f}")

refit_rounds = int(best_iter * 1.15) + 1
refit = xgb.train(PARAMS, dtrval, num_boost_round=refit_rounds, verbose_eval=False)
test_panel['pred'] = refit.predict(dtest)

ics, spreads = daily_ic(test_panel), daily_spread(test_panel)
sec_ics = sector_ic(test_panel)
print(f"Test mean daily IC:          {ics.mean():+.4f}")
print(f"Test within-sector IC:       {sec_ics.mean():+.4f}")
print(f"Test IC information ratio:   {ics.mean() / ics.std():+.2f}")
print(f"Test % days with positive IC: {(ics > 0).mean():.1%}")
print(f"Test top-bottom quintile spread over {HORIZON_DAYS} days: {spreads.mean():+.3%}")

by_sector = test_panel.groupby('sector')['pred'].mean().sort_values(ascending=False)
print(by_sector.round(4).to_string())
print(sec_ics.groupby(level='sector').mean().sort_values(ascending=False)
      .round(4).to_string())

if not save_production:
    print(f"--interactions {args.interactions} / --target {args.target} differ "
          f"from config ({INTERACTION_MODE} / {TARGET_MODE}); "
          "not saving a production model.")
    raise SystemExit

print("Training production model on all data...")
dall = make_dmatrix(panel)
prod_rounds = int(best_iter * 1.3) + 1
prod = xgb.train(PARAMS, dall, num_boost_round=prod_rounds, verbose_eval=False)
prod.save_model('trained_model.json')
print(f"Production model ({prod_rounds} rounds) saved to trained_model.json")

scores = prod.get_score(importance_type='gain')
imp = (pd.Series({f: scores.get(f, 0.0) for f in FEATURE_COLS})
       .sort_values(ascending=False))
print(imp.round(2).to_string())
