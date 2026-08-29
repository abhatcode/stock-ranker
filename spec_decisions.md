MODEL SPECIFICATION AND DESIGN DECISIONS

This file records what changed between versions of the stock ranker, why each
change was made, and how the current model was selected. Written 2026-07-03.


WHAT V1 WAS AND WHY IT FAILED

V1 predicted the raw 30-trading-day return of each stock with XGBoost, trained
on about two years of daily data for roughly 50 tech and defense tickers.
Live validation (see validations.md) showed the June 2 predictions had
negative skill and the June 8 predictions had none. Four defects explained it.

First, the feature set included raw dollar values: the 20, 50 and 200 day
moving averages and MACD were fed in as prices. A tree model splits on
absolute price level, so the model was effectively sorting stocks by share
price instead of by signal. This is why many stocks received byte-identical
predictions and why cheap, volatile names clustered at the top.

Second, the train/test split was done by row order on data whose targets
overlap the next 30 trading days. Rows on either side of the split boundary
shared almost the same forward window, so test performance was inflated by
leakage and the model's real skill was never measured before going live.

Third, predicting raw returns means the dominant driver is overall market
direction, which the model cannot know. Its top picks were simply the
highest-beta stocks, and when the market fell in June they fell roughly three
times as hard.

Fourth, the universe contained tickers that do not exist (SALESFORCE, ARISTA,
INFINIDAT) and the June 2 predictions were generated for stocks the model had
never been trained on.


V2 CHANGES (2026-07-03, MORNING)

- Horizon shortened from 30 to 10 trading days (two calendar weeks), per the
  requirement to predict from a shorter period.
- Every feature made scale-free: returns, ratios and percentile positions
  only, never dollars. This removes the share-price bucketing entirely.
- All features z-scored cross-sectionally within each date, so the model only
  ever sees how a stock compares to the rest of the universe that day. This
  is what makes rankings comparable across categories of stock.
- Target changed from the raw forward return to the percentile rank, within
  each date, of the return in excess of SPY. Subtracting SPY strips out
  market-wide moves; ranking makes the target robust to outliers. Both are
  what cutting through market noise requires.
- Purged time-based split: train, validation and test are separated in time
  with a 10-trading-day gap so overlapping forward windows cannot leak.
- Universe rebuilt as 68 real, liquid tickers across six sectors.

V2 scored a mean daily rank correlation (IC) of +0.037 on held-out data.


V3 CHANGES (2026-07-03, AFTERNOON) - THE CURRENT MODEL

Data. The universe was expanded to 166 liquid large and mid cap names across
ten sectors (semiconductors and hardware, software and internet, healthcare,
financials, energy, materials, consumer staples, consumer discretionary,
industrials and defense, utilities/REITs/telecom), and history was extended
from 4 to 8 years. More stocks per date and more dates are the single most
reliable way to strengthen a cross-sectional model. HES was removed because
it no longer trades (acquired by Chevron).

Features. Grew from 19 to 32, adding effects documented in the asset-pricing
literature: short-term reversal (3-day return), 12-month momentum excluding
the last month, rolling 63-day beta to SPY, idiosyncratic volatility, return
skewness, the maximum daily return of the past month (the lottery effect),
Amihud illiquidity, average overnight gap, and price-volume correlation.
Sector-relative versions of the momentum and volatility features were added
so the model also sees how a stock compares to its own category. In the final
model the new features carry the most importance: 21-day volatility, beta,
12-1 momentum, maximum daily return and skewness are the top gainers.

Model selection. A grid of configurations was trained on the GPU (RTX 4060,
XGBoost with device cuda) and judged strictly on validation-period IC; the
test period was scored only once, at the end, for the selected configuration.
Findings, so they do not get re-tested later:
- A universe-wide rank target beat a sector-neutral target by roughly two to
  one on validation IC (+0.053 versus +0.026). Sector-relative features help,
  but forcing the target itself to be sector-neutral removes real signal.
  The unbiasedness requirement is still met: per-sector average predictions
  span only 0.496 to 0.510 on a 0-to-1 scale.
- Shallow trees won. Depth 2 beat depths 3 and 4 at every setting tried.
- Letting training run longer mattered: the winner uses about 900 boosting
  rounds at learning rate 0.02 and was still improving past the original cap.
- A pairwise learning-to-rank objective (rank:pairwise with per-date groups)
  did much worse than plain regression on the rank target (+0.030 vs +0.058).
- Averaging five random seeds did not improve validation IC, nor did blending
  the depth-2 and depth-4 models or slowing the learning rate to 0.01.

Final configuration: XGBoost regression on the rank target, max_depth 2,
min_child_weight 30, learning_rate 0.02, subsample 0.8, colsample_bytree 0.7,
lambda 5, alpha 0.5, roughly 900 rounds chosen by early stopping.

Honest held-out results, June 2025 through June 2026, a period the model
never saw during training or selection: mean daily IC +0.024, positive on
53.6 percent of days, and the top predicted quintile beat the bottom quintile
by an average of 0.76 percent per 10-trading-day period across 166 stocks.
Validation-period IC was +0.058; the truth likely sits between, and these
honest numbers are far better than v1, whose live IC was negative. Signals
built from price and volume data alone are genuinely weak; anything claiming
a large IC from these inputs is leaking.

The production model is retrained on all available history (through July 2,
2026) with the selected configuration and saved to trained_model.json.
trained_model.pkl is a stale v1 leftover and can be deleted.


VALIDATION SCHEDULE

predictions_v3_20260703.csv was generated from the July 2, 2026 close (July 3
was a market holiday). Its 10-trading-day horizon completes at the close on
Friday, July 17, 2026. To validate, on or after Saturday, July 18: uncomment
the predictions_v3_20260703.csv line in validate_predictions.py and run it.
It will produce validation_results_v3_20260703.csv with the same statistics
used in validations.md. predictions_v2_20260703.csv covers the identical
window and can be validated the same day for a v2-versus-v3 comparison.
What to expect if the model is working: rank IC above zero and the top
quintile beating the bottom quintile. Any single 10-day window is noisy, so
judge the model on several validation cycles, not one.


FILE MAP

- config.py: universe with sector labels, horizon, history length, purge gap
- feature_engineering.py: feature construction, per-date normalization, target
- stock_ranker.py: trains, evaluates honestly, saves the production model
- live_predictions.py: scores the universe today, writes predictions_v3_*.csv
- validate_predictions.py: scores any predictions file against realized prices
- validations.md: validation of the June 8 v1 predictions
- validation_results_*.csv: per-stock output of validate_predictions.py
