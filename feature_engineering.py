"""Builds the model's input features and target from raw price data."""
import json

import numpy as np
import pandas as pd

from config import (HORIZON_DAYS, SECTORS, STOCK_SECTOR,
                    SECTOR_SPECIFIC_FEATURES)

BASE_FEATURES = [
    'ret_3d', 'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'mom_12m_ex1m',
    'excess_ret_21d',
    'vol_21d', 'vol_ratio', 'beta_63', 'idio_vol_63',
    'skew_63', 'max_ret_21d',
    'rsi_14',
    'price_vs_sma20', 'price_vs_sma50', 'sma20_vs_sma50',
    'macd_norm',
    'bollinger_pos', 'price_position_63d', 'dist_from_high_63d',
    'volume_ratio', 'volume_trend', 'amihud_21d', 'pv_corr_21d',
    'overnight_gap_21d',
    'sharpe_21d', 'downside_vol_21d',
]

SECTOR_REL_BASE = ['ret_5d', 'ret_21d', 'ret_63d', 'vol_21d']
SECTOR_REL_FEATURES = [f'sector_rel_{c}' for c in SECTOR_REL_BASE]

SPECIFIC_FEATURES = list(SECTOR_SPECIFIC_FEATURES)
SECTOR_FLAG_COLS = [f'is_{s}' for s in SECTORS]

NORMALIZED_COLS = BASE_FEATURES + SECTOR_REL_FEATURES + SPECIFIC_FEATURES
FEATURE_COLS = NORMALIZED_COLS + SECTOR_FLAG_COLS


def _flatten(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_features(df, stock_name, spy_df=None, with_target=True,
                       macro=None):
    """Compute scale-free features for one stock. Returns a per-date frame.

    macro: optional {ticker: price frame} for the sector-specific features.
    """
    df = _flatten(df)
    sector = STOCK_SECTOR.get(stock_name, 'other')
    close, volume, open_ = df['Close'], df['Volume'], df['Open']
    ret = close.pct_change()

    out = pd.DataFrame(index=df.index)

    out['ret_3d'] = close.pct_change(3)
    out['ret_5d'] = close.pct_change(5)
    out['ret_10d'] = close.pct_change(10)
    out['ret_21d'] = close.pct_change(21)
    out['ret_63d'] = close.pct_change(63)
    out['mom_12m_ex1m'] = close.shift(21) / close.shift(252) - 1

    out['vol_21d'] = ret.rolling(21).std()
    vol_63 = ret.rolling(63).std()
    out['vol_ratio'] = out['vol_21d'] / vol_63
    out['sharpe_21d'] = ret.rolling(21).mean() / ret.rolling(21).std()
    downside = ret.where(ret < 0, 0.0)
    out['downside_vol_21d'] = downside.rolling(21).std()
    out['skew_63'] = ret.rolling(63).skew()
    out['max_ret_21d'] = ret.rolling(21).max()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out['rsi_14'] = 100 - 100 / (1 + gain / loss)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out['price_vs_sma20'] = close / sma20 - 1
    out['price_vs_sma50'] = close / sma50 - 1
    out['sma20_vs_sma50'] = sma20 / sma50 - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out['macd_norm'] = (ema12 - ema26) / close

    bb_std = close.rolling(20).std()
    out['bollinger_pos'] = (close - (sma20 - 2 * bb_std)) / (4 * bb_std)

    low63, high63 = close.rolling(63).min(), close.rolling(63).max()
    out['price_position_63d'] = (close - low63) / (high63 - low63)
    out['dist_from_high_63d'] = close / high63 - 1

    out['volume_ratio'] = volume / volume.rolling(21).mean()
    out['volume_trend'] = volume.rolling(5).mean() / volume.rolling(63).mean()
    dollar_vol = (close * volume).replace(0, np.nan)
    out['amihud_21d'] = (ret.abs() / dollar_vol).rolling(21).mean()
    out['pv_corr_21d'] = ret.rolling(21).corr(volume.pct_change())
    out['overnight_gap_21d'] = (open_ / close.shift(1) - 1).rolling(21).mean()

    if spy_df is not None:
        spy_close = _flatten(spy_df)['Close']
        spy_ret = spy_close.pct_change().reindex(df.index)
        spy_ret_21 = spy_close.pct_change(21).reindex(df.index)
        out['excess_ret_21d'] = out['ret_21d'] - spy_ret_21
        cov = ret.rolling(63).cov(spy_ret)
        var = spy_ret.rolling(63).var()
        out['beta_63'] = cov / var
        out['idio_vol_63'] = (ret - spy_ret).rolling(63).std()
    else:
        out['excess_ret_21d'] = 0.0
        out['beta_63'] = 1.0
        out['idio_vol_63'] = out['vol_21d']

    for feat, (ticker, sectors) in SECTOR_SPECIFIC_FEATURES.items():
        if macro is not None and ticker in macro and sector in sectors:
            m_ret = _flatten(macro[ticker])['Close'].pct_change().reindex(df.index)
            out[feat] = ret.rolling(63).cov(m_ret) / m_ret.rolling(63).var()
        else:
            out[feat] = np.nan

    if with_target:
        fwd = close.shift(-HORIZON_DAYS) / close - 1
        if spy_df is not None:
            spy_close = _flatten(spy_df)['Close']
            spy_fwd = (spy_close.shift(-HORIZON_DAYS) / spy_close - 1).reindex(df.index)
            fwd = fwd - spy_fwd
        out['fwd_excess_return'] = fwd

    out['stock'] = stock_name
    out['sector'] = sector
    for name in SECTORS:
        out[f'is_{name}'] = float(sector == name)
    out['Date'] = out.index
    return out


def build_panel(data, spy_df, with_target=True, macro=None):
    """Stack per-stock feature frames into one panel with sector-relative cols."""
    frames = []
    for stock, df in data.items():
        f = calculate_features(df, stock, spy_df, with_target=with_target,
                               macro=macro)
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=BASE_FEATURES +
                         (['fwd_excess_return'] if with_target else []))

    for col in SECTOR_REL_BASE:
        sector_mean = panel.groupby(['Date', 'sector'])[col].transform('mean')
        panel[f'sector_rel_{col}'] = panel[col] - sector_mean

    return panel.sort_values('Date').reset_index(drop=True)


def cross_sectional_normalize(panel):
    """Z-score each feature within each date, clipped at +/-3 sigma.

    Sector flags are left as 0/1. Sector-specific features are z-scored among
    the stocks they apply to and stay NaN elsewhere (XGBoost handles NaN).
    """
    panel = panel.copy()
    g = panel.groupby('Date')[NORMALIZED_COLS]
    mean, std = g.transform('mean'), g.transform('std')
    panel[NORMALIZED_COLS] = ((panel[NORMALIZED_COLS] - mean) / std.replace(0, np.nan)).clip(-3, 3)
    required = BASE_FEATURES + SECTOR_REL_FEATURES
    return panel.dropna(subset=required)


def interaction_constraints(mode):
    """XGBoost interaction_constraints string for FEATURE_COLS, or None.

    mode 'none': unconstrained. mode 'sector': one group per feature holding
    that feature plus sector flags, so a feature can split by sector but
    never combine with another feature.
    """
    if mode == 'none':
        return None
    if mode != 'sector':
        raise ValueError(f'unknown interaction mode: {mode}')
    idx = {c: i for i, c in enumerate(FEATURE_COLS)}
    groups = []
    for feat in NORMALIZED_COLS:
        if feat in SECTOR_SPECIFIC_FEATURES:
            flags = [f'is_{s}' for s in SECTOR_SPECIFIC_FEATURES[feat][1]]
        else:
            flags = SECTOR_FLAG_COLS
        groups.append([idx[feat]] + [idx[f] for f in flags])
    return json.dumps(groups)


def add_rank_target(panel, mode='universe', blend_weight=0.5):
    """Target = percentile rank (0..1) of the market-excess forward return.

    mode 'universe' ranks against the whole date; 'sector' ranks only within
    the same (date, sector); 'blend' averages the two.
    """
    panel = panel.copy()
    fwd = panel['fwd_excess_return']
    universe = fwd.groupby(panel['Date']).rank(pct=True)
    sector = fwd.groupby([panel['Date'], panel['sector']]).rank(pct=True)
    if mode == 'universe':
        panel['target'] = universe
    elif mode == 'sector':
        panel['target'] = sector
    elif mode == 'blend':
        panel['target'] = blend_weight * universe + (1 - blend_weight) * sector
    else:
        raise ValueError(f'unknown target mode: {mode}')
    return panel
