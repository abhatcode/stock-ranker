"""Feature engineering for the cross-sectional stock ranker (v3).

Same design principles as v2 (every feature scale-free, z-scored per date,
rank target on market-excess return) plus:
- Research-backed anomaly features: short-term reversal, 12-1 momentum,
  rolling beta, idiosyncratic volatility, return skewness, lottery-effect
  MAX, Amihud illiquidity, overnight gaps, price-volume correlation.
- Sector-relative momentum/volatility features and an optional
  sector-neutral target, so the model ranks stocks against their own
  category as well as the whole universe.
"""
import numpy as np
import pandas as pd

from config import HORIZON_DAYS, STOCK_SECTOR

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

# Cross-sectional (per-date) sector-relative versions, added in build_panel.
SECTOR_REL_BASE = ['ret_5d', 'ret_21d', 'ret_63d', 'vol_21d']
SECTOR_REL_FEATURES = [f'sector_rel_{c}' for c in SECTOR_REL_BASE]

FEATURE_COLS = BASE_FEATURES + SECTOR_REL_FEATURES


def _flatten(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_features(df, stock_name, spy_df=None, with_target=True):
    """Compute scale-free features for one stock. Returns a per-date frame."""
    df = _flatten(df)
    close, volume, open_ = df['Close'], df['Volume'], df['Open']
    ret = close.pct_change()

    out = pd.DataFrame(index=df.index)

    # Momentum at several horizons + short-term reversal + classic 12-1
    out['ret_3d'] = close.pct_change(3)
    out['ret_5d'] = close.pct_change(5)
    out['ret_10d'] = close.pct_change(10)
    out['ret_21d'] = close.pct_change(21)
    out['ret_63d'] = close.pct_change(63)
    out['mom_12m_ex1m'] = close.shift(21) / close.shift(252) - 1

    # Volatility and risk-adjusted momentum
    out['vol_21d'] = ret.rolling(21).std()
    vol_63 = ret.rolling(63).std()
    out['vol_ratio'] = out['vol_21d'] / vol_63
    out['sharpe_21d'] = ret.rolling(21).mean() / ret.rolling(21).std()
    downside = ret.where(ret < 0, 0.0)
    out['downside_vol_21d'] = downside.rolling(21).std()
    out['skew_63'] = ret.rolling(63).skew()
    out['max_ret_21d'] = ret.rolling(21).max()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out['rsi_14'] = 100 - 100 / (1 + gain / loss)

    # Trend, expressed relative to price (never in dollars)
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

    # Volume / liquidity / microstructure
    out['volume_ratio'] = volume / volume.rolling(21).mean()
    out['volume_trend'] = volume.rolling(5).mean() / volume.rolling(63).mean()
    dollar_vol = (close * volume).replace(0, np.nan)
    out['amihud_21d'] = (ret.abs() / dollar_vol).rolling(21).mean()
    out['pv_corr_21d'] = ret.rolling(21).corr(volume.pct_change())
    out['overnight_gap_21d'] = (open_ / close.shift(1) - 1).rolling(21).mean()

    # Market-relative risk
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

    if with_target:
        fwd = close.shift(-HORIZON_DAYS) / close - 1
        if spy_df is not None:
            spy_close = _flatten(spy_df)['Close']
            spy_fwd = (spy_close.shift(-HORIZON_DAYS) / spy_close - 1).reindex(df.index)
            fwd = fwd - spy_fwd  # market-excess: strips index-wide noise
        out['fwd_excess_return'] = fwd

    out['stock'] = stock_name
    out['sector'] = STOCK_SECTOR.get(stock_name, 'other')
    out['Date'] = out.index
    return out


def build_panel(data, spy_df, with_target=True):
    """Stack per-stock feature frames into one panel with sector-relative cols."""
    frames = []
    for stock, df in data.items():
        f = calculate_features(df, stock, spy_df, with_target=with_target)
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=BASE_FEATURES +
                         (['fwd_excess_return'] if with_target else []))

    # Sector-relative features: how the stock compares to its own category
    # today. This is what keeps rankings unbiased across sectors.
    for col in SECTOR_REL_BASE:
        sector_mean = panel.groupby(['Date', 'sector'])[col].transform('mean')
        panel[f'sector_rel_{col}'] = panel[col] - sector_mean

    return panel.sort_values('Date').reset_index(drop=True)


def cross_sectional_normalize(panel):
    """Z-score each feature within each date, clipped at +/-3 sigma."""
    panel = panel.copy()
    g = panel.groupby('Date')[FEATURE_COLS]
    mean, std = g.transform('mean'), g.transform('std')
    panel[FEATURE_COLS] = ((panel[FEATURE_COLS] - mean) / std.replace(0, np.nan)).clip(-3, 3)
    return panel.dropna(subset=FEATURE_COLS)


def add_rank_target(panel, sector_neutral=False):
    """Target = percentile rank (0..1) of market-excess forward return per date.

    With sector_neutral=True the forward return is first demeaned within
    (date, sector), so the target only rewards beating your own category --
    the model then cannot express a sector bet at all.
    """
    panel = panel.copy()
    y = panel['fwd_excess_return']
    if sector_neutral:
        y = y - panel.groupby(['Date', 'sector'])['fwd_excess_return'].transform('mean')
        panel['_neutral_fwd'] = y
        panel['target'] = panel.groupby('Date')['_neutral_fwd'].rank(pct=True)
        panel = panel.drop(columns='_neutral_fwd')
    else:
        panel['target'] = panel.groupby('Date')['fwd_excess_return'].rank(pct=True)
    return panel
