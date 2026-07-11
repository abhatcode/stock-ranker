VALIDATION OF THE JUNE 8, 2026 PREDICTIONS

This file scores the predictions saved in predictions_20260608.csv (the v1 model, tech and defense universe) against real market prices. Prices are adjusted closes from yfinance. The window runs from the June 8, 2026 close to the July 2, 2026 close, which is 17 trading days. The model's target horizon was 30 trading days, so this reading covers a little over half of the intended holding period.

The v1 model predicted relative returns (each stock versus the group average), so the statistics that matter are rank correlation (did the stocks ranked higher actually do better) and the outperformance of the top-ranked bucket over the bottom-ranked bucket. Raw return error is not meaningful for a relative ranker.

ACCURACY STATISTICS

- Stocks scored: 52 (PSTG excluded, no longer trades)
- Spearman rank correlation (IC): -0.017 with a p-value of 0.905. Positive would mean the ranking worked; this is statistically indistinguishable from zero.
- Pearson correlation between predicted and actual returns: +0.011
- Directional accuracy versus the group average: 51.9% (50% would be a coin flip)
- Group average return over the window: +2.16%
- Share of all stocks that were positive: 55.8%

OUTPERFORMANCE

- Top 5 predicted stocks returned +3.26% on average, which is +1.10% versus the group. 60% of them were positive.
- Top 10 predicted stocks returned -0.13% on average, -2.30% versus the group, with a 50% win rate.
- Bottom 10 predicted stocks returned +2.24% on average, +0.07% versus the group, with a 50% win rate.
- Bottom 5 predicted stocks returned +1.64% on average.
- Long-short spread (top 5 minus bottom 5): +1.62%

VERDICT

The rankings carried no measurable signal over this window. The rank correlation is effectively zero and the small top-versus-bottom spread is well within what random ordering produces. The model's best calls were badly mixed: its number 8 pick ON fell 24.6% and its number 20 pick ORCL fell 33.8%, while stocks it ranked mid-pack (TENB, PANW, VRNS) were the actual top performers. The causes were identified in the v1 code (dollar-price features and a leaking backtest) and are documented in spec_decisions.md along with the v2 and v3 rewrites.

CHANGES MADE TO THE MODEL IN RESPONSE

The model was rebuilt twice on July 3, 2026 to fix the defects this validation
exposed. The full history and reasoning live in spec_decisions.md; this is the
summary.

Fixes to the failures found here (the v2 rewrite):

- Removed all dollar-priced features. The old model fed raw moving averages
  and MACD in dollars, so its trees split on share price rather than signal.
  That is why so many stocks in the list below share identical predicted
  values. Every feature is now a return, ratio or percentile, never a price.
- Changed the target from the raw 30-trading-day return to the percentile
  rank, within each date, of the return in excess of SPY. Subtracting the
  market strips out index-wide moves the model cannot know, and ranking is
  robust to outliers. This addresses the noise that dominated raw returns.
- Shortened the horizon from 30 trading days to 10.
- Replaced the leaking backtest with a purged time-based split: train,
  validation and test periods are separated by a 10-trading-day gap so
  overlapping forward windows can no longer inflate the results.
- Z-scored every feature across the universe within each date, so the model
  only compares stocks against each other on the same day. Together with a
  universe spanning ten sectors, this removes the category bias.

Further accuracy work (the v3 upgrade, the current model):

- Universe expanded to 166 liquid stocks across ten sectors and training
  history extended to 8 years, giving the model far more cross-sections to
  learn from. Dead tickers were removed.
- Features grew from 19 to 32, adding documented market effects: short-term
  reversal, 12-month momentum excluding the last month, rolling beta,
  idiosyncratic volatility, return skewness, lottery-effect maximum daily
  return, Amihud illiquidity, overnight gaps, and sector-relative versions of
  the momentum and volatility features. These new features carry the most
  importance in the trained model.
- A GPU grid search chose the configuration on validation accuracy alone:
  depth-2 trees, about 900 boosting rounds at learning rate 0.02. A
  sector-neutral target, a pairwise ranking objective and seed ensembles were
  all tested and rejected because they scored worse.

Where the old model showed a rank correlation of -0.017 on this validation,
the rebuilt model scores +0.024 on a full held-out year it never saw, with
the top predicted quintile beating the bottom by 0.76 percent per 10-day
period. Its current predictions are in predictions_v3_20260703.csv, whose
horizon completes at the July 17, 2026 close; validate on or after July 18 by
uncommenting its line in validate_predictions.py and running it.

FULL PER-STOCK COMPARISON

Listed in the model's predicted order, best first. Each line shows the predicted relative return, the actual return over the window, and where the stock actually finished out of the group.

  1. INTC   predicted +0.0205, actual   +9.14%, actual rank 16 of 52
  2. SNOW   predicted +0.0198, actual   +8.19%, actual rank 18 of 52
  3. RKLB   predicted +0.0138, actual  -11.61%, actual rank 46 of 52
  4. DELL   predicted +0.0128, actual   -1.61%, actual rank 35 of 52
  5. UPST   predicted +0.0124, actual  +12.19%, actual rank 9 of 52
  6. NET    predicted +0.0124, actual   -2.17%, actual rank 36 of 52
  7. ZS     predicted +0.0103, actual  +13.99%, actual rank 7 of 52
  8. ON     predicted +0.0096, actual  -24.55%, actual rank 51 of 52
  9. AI     predicted +0.0089, actual  -14.53%, actual rank 47 of 52
 10. LYFT   predicted +0.0079, actual   +9.63%, actual rank 14 of 52
 11. MSTR   predicted +0.0057, actual  -20.78%, actual rank 50 of 52
 12. OKTA   predicted +0.0030, actual  +21.03%, actual rank 5 of 52
 13. TENB   predicted +0.0030, actual  +40.06%, actual rank 1 of 52
 14. MCHP   predicted +0.0029, actual   -7.37%, actual rank 40 of 52
 15. HPE    predicted +0.0029, actual  -17.08%, actual rank 48 of 52
 16. DDOG   predicted +0.0029, actual  +12.38%, actual rank 8 of 52
 17. PANW   predicted +0.0029, actual  +30.69%, actual rank 2 of 52
 18. DASH   predicted +0.0029, actual  +25.92%, actual rank 4 of 52
 19. CRM    predicted +0.0029, actual   -8.77%, actual rank 41 of 52
 20. ORCL   predicted +0.0029, actual  -33.78%, actual rank 52 of 52
 21. PLTR   predicted +0.0029, actual   -5.25%, actual rank 38 of 52
 22. NXPI   predicted +0.0028, actual   -8.92%, actual rank 42 of 52
 23. VRNS   predicted +0.0025, actual  +29.72%, actual rank 3 of 52
 24. CSCO   predicted +0.0025, actual   -9.23%, actual rank 44 of 52
 25. ABNB   predicted +0.0024, actual  +10.79%, actual rank 11 of 52
 26. RTX    predicted +0.0024, actual  +11.52%, actual rank 10 of 52
 27. QCOM   predicted +0.0024, actual  -19.07%, actual rank 49 of 52
 28. NVDA   predicted +0.0024, actual   -6.62%, actual rank 39 of 52
 29. CHKP   predicted +0.0024, actual   +5.65%, actual rank 20 of 52
 30. FTNT   predicted +0.0020, actual   +9.24%, actual rank 15 of 52
 31. UBER   predicted +0.0020, actual   +6.24%, actual rank 19 of 52
 33. TXT    predicted +0.0020, actual   +1.19%, actual rank 29 of 52
 34. PYPL   predicted +0.0020, actual  +10.20%, actual rank 12 of 52
 35. AMZN   predicted -0.0012, actual   -1.04%, actual rank 33 of 52
 36. LRCX   predicted -0.0017, actual   +8.39%, actual rank 17 of 52
 37. COIN   predicted -0.0019, actual   +2.08%, actual rank 27 of 52
 38. BA     predicted -0.0021, actual   +4.90%, actual rank 23 of 52
 39. IBM    predicted -0.0073, actual   +3.10%, actual rank 24 of 52
 40. ADBE   predicted -0.0079, actual  -10.31%, actual rank 45 of 52
 41. AAPL   predicted -0.0108, actual   +2.35%, actual rank 26 of 52
 42. AMD    predicted -0.0138, actual   +5.61%, actual rank 21 of 52
 43. GOOG   predicted -0.0166, actual   -1.38%, actual rank 34 of 52
 44. MSFT   predicted -0.0166, actual   -5.16%, actual rank 37 of 52
 45. MU     predicted -0.0166, actual   +2.77%, actual rank 25 of 52
 46. LMT    predicted -0.0179, actual   +4.97%, actual rank 22 of 52
 47. NOC    predicted -0.0179, actual   +1.52%, actual rank 28 of 52
 48. GD     predicted -0.0179, actual  +10.07%, actual rank 13 of 52
 49. HII    predicted -0.0179, actual   -0.26%, actual rank 31 of 52
 50. META   predicted -0.0183, actual   -0.33%, actual rank 32 of 52
 51. CRWD   predicted -0.0183, actual  +17.78%, actual rank 6 of 52
 52. AVGO   predicted -0.0203, actual   -8.97%, actual rank 43 of 52
 53. SSNLF  predicted -0.0985, actual   +0.00%, actual rank 30 of 52

Generated 2026-07-03 from validation_results_20260608.csv.
