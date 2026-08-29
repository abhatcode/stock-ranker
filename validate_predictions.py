"""Validate saved prediction CSVs against realized returns.

The model's target is the cross-sectionally demeaned 30-trading-day forward
return, so the meaningful checks are ranking quality (Spearman IC), the
top-vs-bottom bucket spread, and demeaned directional accuracy -- not raw
return error.
"""
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr

# prediction file -> date the predictions were made (entry close).
# The v3 file's 10-trading-day horizon completes at the 2026-07-17 close;
# add it here and run this script on/after 2026-07-18.
PREDICTION_FILES = {
    'predictions_20260602.csv': '2026-06-02',
    'predictions_20260608.csv': '2026-06-08',
    'predictions_v2_20260703.csv': '2026-07-02',
    'predictions_v3_20260703.csv': '2026-07-02',
}


def validate(pred_file, pred_date):
    preds = pd.read_csv(pred_file)
    if 'score' in preds.columns:  # v3+ files call the prediction 'score'
        preds = preds.rename(columns={'score': 'predicted_return'})
    tickers = preds['stock'].tolist()

    print(f"\n{'=' * 70}")
    print(f"VALIDATING {pred_file}  (prediction date: {pred_date})")
    print(f"{'=' * 70}")

    px = yf.download(tickers, start=pred_date, progress=False, auto_adjust=True)['Close']
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px.dropna(axis=1, how='all')
    px = px.dropna(axis=0, how='all')

    entry_date = px.index[0]
    exit_date = px.index[-1]
    n_days = len(px.index) - 1
    print(f"Entry close: {entry_date.date()}   Latest close: {exit_date.date()}   ({n_days} trading days elapsed)")

    entry = px.iloc[0]
    exit_ = px.iloc[-1]
    actual = ((exit_ - entry) / entry).rename('actual_return')

    df = preds.merge(actual, left_on='stock', right_index=True, how='left')
    missing = df[df['actual_return'].isna()]['stock'].tolist()
    if missing:
        print(f"No price data for: {missing} (excluded)")
    df = df.dropna(subset=['actual_return']).copy()

    # Demean actuals to match the model's relative-return target
    df['actual_vs_group'] = df['actual_return'] - df['actual_return'].mean()
    df['actual_rank'] = df['actual_return'].rank(ascending=False).astype(int)

    ic, ic_p = spearmanr(df['predicted_return'], df['actual_return'])
    pearson = df['predicted_return'].corr(df['actual_return'])

    n = len(df)
    k = max(5, n // 10)
    top = df.nsmallest(k, 'rank')       # best predicted (rank 1 = best)
    bottom = df.nlargest(k, 'rank')

    dir_acc = ((df['predicted_return'] > 0) == (df['actual_vs_group'] > 0)).mean()

    print(f"\nStocks evaluated: {n}")
    print(f"Spearman rank IC:            {ic:+.3f}  (p-value {ic_p:.3f})")
    print(f"Pearson correlation:         {pearson:+.3f}")
    print(f"Directional accuracy (vs group mean): {dir_acc:.1%}")
    print(f"\nGroup average actual return: {df['actual_return'].mean():+.2%}")
    print(f"Top {k} predicted   -> avg actual: {top['actual_return'].mean():+.2%}")
    print(f"Bottom {k} predicted -> avg actual: {bottom['actual_return'].mean():+.2%}")
    print(f"Long-short spread (top - bottom):   {top['actual_return'].mean() - bottom['actual_return'].mean():+.2%}")

    show = df.sort_values('rank')[['rank', 'stock', 'predicted_return', 'actual_return', 'actual_rank']]
    show['predicted_return'] = show['predicted_return'].map('{:+.4f}'.format)
    show['actual_return'] = show['actual_return'].map('{:+.2%}'.format)
    print(f"\nFull comparison (predicted rank order):")
    print(show.to_string(index=False))

    out = pred_file.replace('predictions', 'validation_results')
    df.to_csv(out, index=False)
    print(f"\nSaved detailed results to {out}")
    return {'file': pred_file, 'n': n, 'ic': ic, 'p': ic_p, 'dir_acc': dir_acc,
            'spread': top['actual_return'].mean() - bottom['actual_return'].mean(),
            'days': n_days}


if __name__ == '__main__':
    summaries = [validate(f, d) for f, d in PREDICTION_FILES.items()]
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for s in summaries:
        print(f"{s['file']}: n={s['n']}, {s['days']} trading days elapsed, "
              f"IC={s['ic']:+.3f} (p={s['p']:.3f}), long-short spread={s['spread']:+.2%}, "
              f"dir acc={s['dir_acc']:.1%}")
