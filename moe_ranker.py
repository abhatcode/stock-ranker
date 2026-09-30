"""Trains the gated mixture of sector experts (v6). See moe.py and the README.

Fits experts and a supervisor on the train dates, grids the conductor
(expert kind, confidence gating, supervisor weight) on validation, refits
on train+val to score the test period once, then fits on all data and
saves to models/ for live_predictions.py --arch moe.

    python moe_ranker.py --data-cache raw.pkl
"""
import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta

from config import (STOCKS, BENCHMARK, HORIZON_DAYS, HISTORY_DAYS, PURGE_DAYS,
                    MACRO_TICKERS, SECTORS)
from feature_engineering import (FEATURE_COLS, build_panel,
                                 cross_sectional_normalize, add_rank_target,
                                 interaction_constraints)
from metrics import daily_ic, sector_ic, daily_spread, group_ic
from moe import (SUPERVISOR_COLS, expert_features, supervisor_frame, dmatrix,
                 predict_experts, confidence, combine, save_moe)

parser = argparse.ArgumentParser()
parser.add_argument('--data-cache',
                    help='pickle of the raw download; reused if it exists')
args = parser.parse_args()

BASE_PARAMS = {
    'objective': 'rank:pairwise',
    'eval_metric': 'auc',
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
SHARED_PARAMS = {**BASE_PARAMS,
                 'interaction_constraints': interaction_constraints('sector')}
EXPERT_PARAMS = BASE_PARAMS
SUPERVISOR_PARAMS = BASE_PARAMS
MAX_ROUNDS = 6000
EARLY_STOP = 150
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

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
panel = add_rank_target(panel, mode='sector')
panel = panel.sort_values(['Date', 'sector'], kind='stable').reset_index(drop=True)
sup = supervisor_frame(panel, spy_data, macro)
print(f"Panel: {len(panel)} rows, {panel['Date'].nunique()} dates; "
      f"supervisor: {len(sup)} (date, sector) rows")

dates = np.array(sorted(panel['Date'].unique()))
n = len(dates)
train_end, val_end = int(n * 0.70), int(n * 0.85)

train_dates = dates[:train_end - PURGE_DAYS]
val_dates = dates[train_end:val_end - PURGE_DAYS]
test_dates = dates[val_end:]
trval_dates = dates[:val_end - PURGE_DAYS]


def by_dates(frame, date_arr):
    return frame[frame['Date'].isin(date_arr)].copy()


print(f"Train {len(train_dates)} dates ({train_dates[0].date()} .. {train_dates[-1].date()})")
print(f"Val   {len(val_dates)} dates ({val_dates[0].date()} .. {val_dates[-1].date()})")
print(f"Test  {len(test_dates)} dates ({test_dates[0].date()} .. {test_dates[-1].date()})")


def fit(params, cols, keys, train, val=None, rounds=None):
    """Early-stop on val when given, else train a fixed number of rounds."""
    dtrain = dmatrix(train, cols, keys)
    if val is None:
        return xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False), rounds
    b = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                  evals=[(dmatrix(val, cols, keys), 'val')],
                  early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    best = b.best_iteration + 1
    return b[:best], best


def fit_experts(kind, train, val=None, rounds=None):
    """kind 'shared' -> {'shared': model}; 'per_sector' -> {sector: model}."""
    if kind == 'shared':
        b, r = fit(SHARED_PARAMS, FEATURE_COLS, ['Date', 'sector'], train, val,
                   rounds and rounds['shared'])
        return {'shared': b}, {'shared': r}
    experts, used = {}, {}
    for s in SECTORS:
        tr = train[train['sector'] == s]
        va = None if val is None else val[val['sector'] == s]
        experts[s], used[s] = fit(EXPERT_PARAMS, expert_features(s), ['Date'],
                                  tr, va, rounds and rounds[s])
    return experts, used


def scaled(rounds, factor):
    return {k: int(v * factor) + 1 for k, v in rounds.items()}


def sup_predict(model, frame):
    return model.predict(xgb.DMatrix(frame[SUPERVISOR_COLS]))


print("Fitting experts and supervisor on train (early stopping on val AUC)...")
train, val = by_dates(panel, train_dates), by_dates(panel, val_dates)
sup_train, sup_val = by_dates(sup, train_dates), by_dates(sup, val_dates)

expert_rounds = {}
for kind in ['shared', 'per_sector']:
    experts, expert_rounds[kind] = fit_experts(kind, train, val)
    val[f'pred_{kind}'] = predict_experts(experts, val)
    print(f"{kind} rounds {expert_rounds[kind]}")

supervisor, sup_rounds = fit(SUPERVISOR_PARAMS, SUPERVISOR_COLS, ['Date'], sup_train, sup_val)
sup_val['sup'] = sup_predict(supervisor, sup_val)
print(f"supervisor rounds {sup_rounds}")

print(f"v5 raw score (bar to beat), universe IC: {daily_ic(val, 'pred_shared').mean():+.4f}")
for kind in ['shared', 'per_sector']:
    print(f"{kind} experts, within-sector IC: {sector_ic(val, f'pred_{kind}').mean():+.4f}")
print(f"Supervisor sector-level IC (10 sectors/day): "
      f"{group_ic(sup_val, ['Date'], min_size=10, pred='sup').mean():+.4f}")

grid = []
for kind in ['shared', 'per_sector']:
    conf = confidence(val, f'pred_{kind}')
    for use_conf in [False, True]:
        for lam in LAMBDAS:
            val['score'] = combine(val, f'pred_{kind}', sup_val, 'sup', lam,
                                   conf if use_conf else None)
            grid.append({'experts': kind, 'confidence': use_conf, 'lambda': lam,
                         'val_ic': daily_ic(val, 'score').mean(),
                         'val_in_sector_ic': sector_ic(val, 'score').mean(),
                         'val_spread': daily_spread(val, 'score').mean()})
grid = pd.DataFrame(grid).sort_values('val_ic', ascending=False).reset_index(drop=True)
print(grid.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))

best = grid.iloc[0]
kind, use_conf, lam = best['experts'], bool(best['confidence']), float(best['lambda'])
print(f"Selected: experts={kind}, confidence={use_conf}, lambda={lam}")

trval, test = by_dates(panel, trval_dates), by_dates(panel, test_dates)
sup_trval, sup_test = by_dates(sup, trval_dates), by_dates(sup, test_dates)

refit_experts, _ = fit_experts(kind, trval, rounds=scaled(expert_rounds[kind], 1.15))
refit_sup, _ = fit(SUPERVISOR_PARAMS, SUPERVISOR_COLS, ['Date'], sup_trval,
                   rounds=int(sup_rounds * 1.15) + 1)
test['pred'] = predict_experts(refit_experts, test)
sup_test['sup'] = sup_predict(refit_sup, sup_test)

test_conf = None
if use_conf:
    hist = pd.concat([val.assign(pred=val[f'pred_{kind}']), test])
    hist = hist[['Date', 'sector', 'fwd_excess_return', 'pred']]
    hist = hist.sort_values(['Date', 'sector'], kind='stable')
    hist['conf'] = confidence(hist, 'pred')
    test_conf = hist.loc[test.index, 'conf']
test['score'] = combine(test, 'pred', sup_test, 'sup', lam, test_conf)

ics, spreads = daily_ic(test, 'score'), daily_spread(test, 'score')
sec_ics = sector_ic(test, 'score')
print(f"Test mean daily IC:          {ics.mean():+.4f}")
print(f"Test within-sector IC:       {sec_ics.mean():+.4f}")
print(f"Test IC information ratio:   {ics.mean() / ics.std():+.2f}")
print(f"Test % days with positive IC: {(ics > 0).mean():.1%}")
print(f"Test top-bottom quintile spread over {HORIZON_DAYS} days: {spreads.mean():+.3%}")
print(f"Supervisor sector-level IC:  "
      f"{group_ic(sup_test, ['Date'], min_size=10, pred='sup').mean():+.4f}")
if use_conf:
    print(f"Share of (date, sector) rows dialled down: {(test_conf < 1).mean():.1%}")

print(test.groupby('sector')['score'].mean().sort_values(ascending=False)
      .round(3).to_string())
print(sec_ics.groupby(level='sector').mean().sort_values(ascending=False)
      .round(4).to_string())

print("Training production models on all data...")
prod_experts, _ = fit_experts(kind, panel, rounds=scaled(expert_rounds[kind], 1.3))
prod_sup, _ = fit(SUPERVISOR_PARAMS, SUPERVISOR_COLS, ['Date'], sup,
                  rounds=int(sup_rounds * 1.3) + 1)
conductor = {'experts': kind, 'lambda': lam, 'confidence': use_conf,
             'trained_through': str(pd.Timestamp(dates[-1]).date())}
save_moe(prod_experts, prod_sup, conductor)
if use_conf:
    save_moe(refit_experts, refit_sup, None, prefix='monitor_')
print(f"Saved models/ with conductor {conductor}")

scores = prod_sup.get_score(importance_type='gain')
print(pd.Series({f: scores.get(f, 0.0) for f in SUPERVISOR_COLS})
      .sort_values(ascending=False).head(12).round(2).to_string())
