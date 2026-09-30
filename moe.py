"""Gated mixture of sector experts (v6). Design details are in the README.

Used by moe_ranker.py (training) and live_predictions.py (scoring).
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from config import HORIZON_DAYS, SECTORS, SECTOR_SPECIFIC_FEATURES
from feature_engineering import (FEATURE_COLS, NORMALIZED_COLS,
                                 SECTOR_FLAG_COLS, _flatten)
from metrics import sector_ic

MODEL_DIR = 'models'

SECTOR_AGG_BASE = ['ret_5d', 'ret_21d', 'ret_63d', 'mom_12m_ex1m', 'vol_21d',
                   'beta_63', 'dist_from_high_63d', 'volume_trend', 'amihud_21d']
SECTOR_AGG_COLS = [f'sec_{c}' for c in SECTOR_AGG_BASE] + ['sec_disp_ret_21d']
REGIME_COLS = ['mkt_ret_21d', 'mkt_ret_63d', 'mkt_vol_21d', 'mkt_vol_ratio',
               'mkt_drawdown_63d', 'tlt_ret_21d', 'uso_ret_21d']
SUPERVISOR_COLS = SECTOR_AGG_COLS + REGIME_COLS + SECTOR_FLAG_COLS


def expert_features(sector):
    """Stock features for one sector's expert (drops betas that are NaN there)."""
    return [c for c in NORMALIZED_COLS
            if c not in SECTOR_SPECIFIC_FEATURES
            or sector in SECTOR_SPECIFIC_FEATURES[c][1]]


def market_regime(spy_df, macro):
    """Per-date market-condition features."""
    spy = _flatten(spy_df)['Close']
    ret = spy.pct_change()
    out = pd.DataFrame(index=spy.index)
    out['mkt_ret_21d'] = spy.pct_change(21)
    out['mkt_ret_63d'] = spy.pct_change(63)
    out['mkt_vol_21d'] = ret.rolling(21).std()
    out['mkt_vol_ratio'] = out['mkt_vol_21d'] / ret.rolling(126).std()
    out['mkt_drawdown_63d'] = spy / spy.rolling(63).max() - 1
    for ticker, col in [('TLT', 'tlt_ret_21d'), ('USO', 'uso_ret_21d')]:
        if macro is not None and ticker in macro:
            close = _flatten(macro[ticker])['Close']
            out[col] = close.pct_change(21).reindex(spy.index)
        else:
            out[col] = np.nan
    return out


def supervisor_frame(panel, spy_df, macro):
    """One row per (Date, sector) for the supervisor."""
    g = panel.groupby(['Date', 'sector'])
    sec = g[SECTOR_AGG_BASE].mean().add_prefix('sec_')
    sec['sec_disp_ret_21d'] = g['ret_21d'].std()
    has_target = 'fwd_excess_return' in panel.columns
    if has_target:
        sec['fwd_excess_return'] = g['fwd_excess_return'].mean()
    sec = sec.reset_index()
    sec = sec.merge(market_regime(spy_df, macro), left_on='Date',
                    right_index=True, how='left')
    for name in SECTORS:
        sec[f'is_{name}'] = (sec['sector'] == name).astype(float)
    if has_target:
        sec['target'] = sec.groupby('Date')['fwd_excess_return'].rank(pct=True)
    return sec.sort_values(['Date', 'sector'], kind='stable').reset_index(drop=True)


def dmatrix(part, cols, keys):
    """Ranking DMatrix with one query group per unique value of keys."""
    qid = part.groupby(keys, sort=True).ngroup().to_numpy()
    label = part['target'] if 'target' in part.columns else None
    return xgb.DMatrix(part[cols], label=label, qid=qid)


def predict_experts(experts, panel):
    """Raw expert scores for every row of a (Date, sector)-sorted panel."""
    if 'shared' in experts:
        return pd.Series(experts['shared'].predict(xgb.DMatrix(panel[FEATURE_COLS])),
                         index=panel.index)
    pred = pd.Series(np.nan, index=panel.index)
    for sector, booster in experts.items():
        mask = panel['sector'] == sector
        if mask.any():
            pred[mask] = booster.predict(
                xgb.DMatrix(panel.loc[mask, expert_features(sector)]))
    return pred


def within_z(panel, col):
    """z-score of col within each (Date, sector)."""
    g = panel.groupby(['Date', 'sector'])[col]
    return ((panel[col] - g.transform('mean'))
            / g.transform('std').replace(0, np.nan)).fillna(0.0)


def _recent_ic(ic, lookback):
    return ic.rolling(lookback, min_periods=21).mean()


def confidence(panel, pred_col, lookback=63, floor=0.5):
    """Per-row expert multiplier from recent realized within-sector IC."""
    ic = sector_ic(panel, pred=pred_col).unstack('sector')
    ic = ic.reindex(sorted(panel['Date'].unique()))
    # Shift by the horizon: the IC of date d is only known once its forward
    # window closes, so today's multiplier can't use today's own result.
    known = _recent_ic(ic.shift(HORIZON_DAYS), lookback)
    mult = pd.DataFrame(np.where(known < 0, floor, 1.0),
                        index=known.index, columns=known.columns)
    mult = mult.rename_axis(index='Date', columns='sector').stack().rename('conf')
    rows = panel[['Date', 'sector']].merge(mult.reset_index(), on=['Date', 'sector'],
                                           how='left')
    return pd.Series(rows['conf'].fillna(1.0).to_numpy(), index=panel.index)


def current_confidence(history, pred_col, lookback=63, floor=0.5):
    """Live version of confidence(): one multiplier per sector for today.

    history must hold only dates whose forward window has already closed.
    """
    ic = sector_ic(history, pred=pred_col).unstack('sector')
    ic = ic.reindex(sorted(history['Date'].unique()))
    recent = _recent_ic(ic, lookback).iloc[-1]
    return pd.Series(np.where(recent < 0, floor, 1.0), index=recent.index)


def combine(panel, expert_col, sup, sup_col, lam, conf=None):
    """Conductor: confidence * within-sector z + lam * per-date sector z."""
    z = within_z(panel, expert_col)
    if conf is not None:
        z = z * conf
    g = sup.groupby('Date')[sup_col]
    sup_z = ((sup[sup_col] - g.transform('mean'))
             / g.transform('std').replace(0, np.nan)).fillna(0.0)
    tilt = sup[['Date', 'sector']].assign(tilt=sup_z)
    rows = panel[['Date', 'sector']].merge(tilt, on=['Date', 'sector'], how='left')
    return z + lam * pd.Series(rows['tilt'].fillna(0.0).to_numpy(), index=panel.index)


def save_moe(experts, supervisor, conductor, prefix='', directory=MODEL_DIR):
    os.makedirs(directory, exist_ok=True)
    for name, booster in experts.items():
        booster.save_model(os.path.join(directory, f'{prefix}expert_{name}.json'))
    supervisor.save_model(os.path.join(directory, f'{prefix}supervisor.json'))
    if conductor is not None:
        with open(os.path.join(directory, 'conductor.json'), 'w') as f:
            json.dump(conductor, f, indent=2)


def load_moe(prefix='', directory=MODEL_DIR):
    with open(os.path.join(directory, 'conductor.json')) as f:
        conductor = json.load(f)
    names = ['shared'] if conductor['experts'] == 'shared' else list(SECTORS)
    experts = {}
    for name in names:
        b = xgb.Booster()
        b.load_model(os.path.join(directory, f'{prefix}expert_{name}.json'))
        experts[name] = b
    supervisor = xgb.Booster()
    supervisor.load_model(os.path.join(directory, f'{prefix}supervisor.json'))
    return experts, supervisor, conductor
