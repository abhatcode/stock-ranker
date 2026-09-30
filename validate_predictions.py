"""Checks a saved predictions CSV against what actually happened."""
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr

from config import STOCK_SECTOR

PREDICTION_FILES = {
    'predictions_20260602.csv': '2026-06-02',
    'predictions_20260608.csv': '2026-06-08',
    'predictions_v2_20260703.csv': '2026-07-02',
    'predictions_v3_20260703.csv': '2026-07-02',
    # v4/v5/v6 horizon ends 2026-10-14; uncomment after 2026-10-15.
    # 'predictions_v4_20260930.csv': '2026-09-30',
    # 'predictions_v5_20260930.csv': '2026-09-30',
    # 'predictions_v6_20260930.csv': '2026-09-30',
}


def validate(pred_file, pred_date):
    preds = pd.read_csv(pred_file)
    if 'score' in preds.columns:
        preds = preds.rename(columns={'score': 'predicted_return'})
    tickers = preds['stock'].tolist()

    print(f"validating {pred_file} (prediction date: {pred_date})")

    px = yf.download(tickers, start=pred_date, progress=False, auto_adjust=True)['Close']
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px.dropna(axis=1, how='all')
    px = px.dropna(axis=0, how='all')

    entry_date = px.index[0]
    exit_date = px.index[-1]
    n_days = len(px.index) - 1
    print(f"entry close {entry_date.date()}, latest close {exit_date.date()}, {n_days} trading days elapsed")

    entry = px.iloc[0]
    exit_ = px.iloc[-1]
    actual = ((exit_ - entry) / entry).rename('actual_return')

    df = preds.merge(actual, left_on='stock', right_index=True, how='left')
    missing = df[df['actual_return'].isna()]['stock'].tolist()
    if missing:
        print(f"no price data for: {missing} (excluded)")
    df = df.dropna(subset=['actual_return']).copy()

    df['actual_vs_group'] = df['actual_return'] - df['actual_return'].mean()
    df['actual_rank'] = df['actual_return'].rank(ascending=False).astype(int)

    ic, ic_p = spearmanr(df['predicted_return'], df['actual_return'])
    pearson = df['predicted_return'].corr(df['actual_return'])

    n = len(df)
    k = max(5, n // 10)
    top = df.nsmallest(k, 'rank')
    bottom = df.nlargest(k, 'rank')

    dir_acc = ((df['predicted_return'] > 0) == (df['actual_vs_group'] > 0)).mean()

    df['sector'] = df['stock'].map(STOCK_SECTOR)
    per_sector = [spearmanr(g['predicted_return'], g['actual_return'])[0]
                  for _, g in df.groupby('sector') if len(g) >= 5]
    sector_ic = np.nanmean(per_sector) if per_sector else np.nan

    print(f"stocks evaluated: {n}")
    print(f"rank IC: {ic:+.3f} (p={ic_p:.3f})")
    print(f"pearson correlation: {pearson:+.3f}")
    print(f"within-sector rank IC: {sector_ic:+.3f}")
    print(f"directional accuracy vs group mean: {dir_acc:.1%}")
    print(f"group average actual return: {df['actual_return'].mean():+.2%}")
    print(f"top {k} predicted, avg actual: {top['actual_return'].mean():+.2%}")
    print(f"bottom {k} predicted, avg actual: {bottom['actual_return'].mean():+.2%}")
    print(f"long-short spread: {top['actual_return'].mean() - bottom['actual_return'].mean():+.2%}")

    show = df.sort_values('rank')[['rank', 'stock', 'predicted_return', 'actual_return', 'actual_rank']]
    show['predicted_return'] = show['predicted_return'].map('{:+.4f}'.format)
    show['actual_return'] = show['actual_return'].map('{:+.2%}'.format)
    print(show.to_string(index=False))

    out = pred_file.replace('predictions', 'validation_results')
    df.to_csv(out, index=False)
    print(f"saved detailed results to {out}")
    return {'file': pred_file, 'n': n, 'ic': ic, 'p': ic_p, 'dir_acc': dir_acc,
            'spread': top['actual_return'].mean() - bottom['actual_return'].mean(),
            'days': n_days, 'sector_ic': sector_ic}


if __name__ == '__main__':
    summaries = [validate(f, d) for f, d in PREDICTION_FILES.items()]
    for s in summaries:
        print(f"{s['file']}: n={s['n']}, {s['days']} trading days elapsed, "
              f"IC={s['ic']:+.3f} (p={s['p']:.3f}), sector IC={s['sector_ic']:+.3f}, long-short spread={s['spread']:+.2%}, "
              f"dir acc={s['dir_acc']:.1%}")
