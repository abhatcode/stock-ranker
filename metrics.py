"""Vectorized ranking metrics shared by the training scripts."""
import numpy as np
import pandas as pd


def group_ic(part, keys, min_size, pred='pred', actual='fwd_excess_return'):
    """Spearman rank correlation between pred and actual within each group."""
    by = [part[k] for k in keys]
    r = part[[pred, actual]].groupby(by).rank()
    d = r - r.groupby(by).transform('mean')
    sums = pd.DataFrame({
        'xy': d[pred] * d[actual],
        'xx': d[pred] ** 2,
        'yy': d[actual] ** 2,
        'n': 1,
    }).groupby(by).sum()
    sums = sums[sums['n'] >= min_size]
    return (sums['xy'] / np.sqrt(sums['xx'] * sums['yy'])).fillna(0.0)


def daily_ic(part, pred='pred'):
    return group_ic(part, ['Date'], min_size=20, pred=pred)


def sector_ic(part, pred='pred'):
    return group_ic(part, ['Date', 'sector'], min_size=5, pred=pred)


def daily_spread(part, pred='pred'):
    """Top-minus-bottom quintile forward return per date."""
    g = part.groupby('Date')[pred]
    pos = g.rank(ascending=False, method='first')
    size = g.transform('size')
    k = np.maximum(5, size // 5)
    fwd = part['fwd_excess_return']
    top = fwd.where(pos <= k).groupby(part['Date']).mean()
    bottom = fwd.where(pos > size - k).groupby(part['Date']).mean()
    return (top - bottom)[part.groupby('Date').size() >= 20]
