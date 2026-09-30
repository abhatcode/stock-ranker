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


V3 CHANGES (2026-07-03, AFTERNOON)

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


V4 CHANGES (2026-09-30, AFTERNOON)

Motivation. v3 lets every feature combine with every other feature in one
global rule applied to all 166 stocks. A metric that only matters in one
industry (the classic example is inventory turnover: vital for a wholesaler,
irrelevant to software) then adds noise to every other industry's ranking.

What changed.
- Ten one-hot sector flags (is_energy, is_financials, ...) were added as
  features. They are 0/1 and are not z-scored.
- XGBoost interaction_constraints (config.INTERACTION_MODE = 'sector'): each
  feature sits in its own group together with the sector flags. A tree
  branch may only combine features from one group, so a feature's effect can
  differ by sector, but two features can never be combined into a cross-
  feature rule that is applied to every stock.
- Sector-specific features. The model has no fundamentals (yfinance has no
  point-in-time fundamental history, so inventory turnover and similar
  ratios cannot be trained on honestly). Two price-derived examples were added
  instead: rate_beta_63 (63-day beta to TLT) for financials and
  utilities/REITs/telecom, and oil_beta_63 (beta to USO) for energy. They are
  NaN for every other stock and may only interact with their own sectors'
  flags, so they can never change the relative order of stocks outside
  those sectors. Add more in config.SECTOR_SPECIFIC_FEATURES.
- stock_ranker.py now prints validation-period IC and takes
  --interactions none|sector for A/B runs. Only the config default saves the
  production model.

A/B result, same data and splits (train Oct 2019 to Jul 2024, val Aug 2024
to Aug 2025, test Aug 2025 to Sep 2026), everything else identical:
- Unconstrained: val IC +0.041, test IC +0.028, test quintile spread
  +1.24 percent per 10 days, 214 rounds.
- Sector constrained: val IC +0.048, test IC +0.045, positive on 55.5
  percent of test days, test quintile spread +2.08 percent per 10 days,
  85 rounds.
The constrained model won on validation IC (the selection rule) and also on
the one-time test score, so it was kept. These numbers are not comparable to
the v3 figures above, which used a window ending three months earlier.

Caveats.
- Early stopping picked only 85 rounds, and the production model (111
  rounds) uses just 11 of the 34 non-flag features plus the sector flags.
  The two sector-specific betas got zero importance, so they are plumbing
  for future features, not a source of the gain.
- Because a flag can now split on its own, the model can make outright
  sector bets. Test-period per-sector average predictions stayed in a narrow
  0.495 to 0.515 band (similar to unconstrained), but the first v4 live list
  (2026-09-30) put 16 semis/hardware names in its top 20. Watch this.


V5 CHANGES (2026-09-30, EVENING) - THE CURRENT MODEL

Goal. Teach the model what makes a top tech stock a top tech stock, instead
of only ranking all 165 stocks against each other on each date.

Is "group by date plus sector with query ids" applicable? Only partly.
- XGBoost query ids (qid) are a single integer per row, not a multi-level
  key, and rows must be sorted by qid. A (date, sector) group is made by
  numbering each (date, sector) pair and sorting the panel by date then
  sector. That part works.
- qid is ignored by regression objectives. v3 and v4 use reg:squarederror,
  so adding qid to them does nothing. Groups only matter for a ranking
  objective (rank:pairwise, rank:ndcg, rank:map).
- A ranking loss only compares stocks inside the same group, so with
  (date, sector) groups the model never learns how one sector should rank
  against another. Where a whole sector lands in the universe list is not
  trained.
So it was implemented both ways and compared: literally, as rank:pairwise
with (date, sector) query groups, and as regression equivalents that need no
qid (a target ranked within each date and sector, and a 50/50 blend of the
universe and within-sector ranks).

What changed.
- config.TARGET_MODE picks one of: universe, sector_rank, blend,
  pairwise_sector. stock_ranker.py takes --target to A/B them and
  --data-cache FILE so every run uses the identical download.
- Evaluation is vectorized: IC per date and per (date, sector) is computed
  with grouped rank sums instead of one scipy call per group, and it matches
  scipy to 1e-16. Both universe IC and within-sector IC are reported, plus
  within-sector IC broken down by sector.
- Each mode early-stops on its own loss: RMSE for the regression modes, and
  for pairwise the share of correctly ordered pairs inside each group
  (ranking AUC). Early stopping directly on validation IC was tried first and
  rejected: it picked a one-tree model (a single split on 21-day
  volatility), a lucky low-volatility sort rather than a model.
- live_predictions.py adds sector and sector_rank columns and prints the top
  3 in each sector. Pairwise scores are unscaled, so the score column is
  converted to a universe percentile.
- validate_predictions.py also reports within-sector rank IC. For reference,
  v3's live July file scored +0.135 within sectors.

A/B results, same cached data and splits, sector interaction constraints on
in every run. "In-sector" means the average IC inside each (date, sector)
group.
- universe (v4 behaviour): val IC +0.048, val in-sector +0.003; test IC
  +0.048, test in-sector +0.030, quintile spread +2.18 percent, 70 rounds.
- sector_rank regression: val IC -0.001, val in-sector -0.000; stopped
  after 16 rounds. Failed.
- blend regression: val IC +0.015, val in-sector +0.022; stopped after
  1 round. Failed.
- pairwise_sector: val IC +0.058, val in-sector +0.037; test IC +0.044,
  test in-sector +0.049, quintile spread +1.65 percent, 79 rounds.
pairwise_sector won both validation measures, so it is the production model
(103 rounds). On the one-time test score it was clearly better within
sectors and slightly worse on the universe-wide ranking and quintile spread.

Caveats.
- The two regression variants hit early stopping almost immediately
  because RMSE on a within-sector rank barely moves. They may deserve a
  retry with a different stopping rule, but on this evidence the pairwise
  form is the one that learns within-sector order.
- The sector-specific betas (rate_beta_63, oil_beta_63) went from zero
  importance in v4 to mid-table in v5. They carry information about which
  bank or which oil producer to prefer, not about banks versus software.
- Cross-sector placement is untrained (see above). The first v5 list
  (2026-09-30) put 7 of its bottom 10 in energy. Use sector_rank when the
  question is "best stock in this sector"; the universe rank mixes that with
  untrained sector placement.


V6 GATED SECTOR EXPERTS (2026-09-30, NIGHT) - RUNNING IN SHADOW, NOT PRODUCTION

Idea. v5 ranks well inside sectors but never learns how sectors compare to
each other. v6 adds a supervisor for exactly that, on top of sector
experts:
- Experts rank stocks inside one sector: either ten separate per-sector
  XGBoost models or the single shared v5 model.
- A supervisor ranks the ten sectors against each other on each date. Its
  inputs are sector averages (momentum, volatility, beta, distance from
  high, liquidity, volume trend, dispersion), market-regime features (SPY
  21 and 63 day return, SPY volatility and its ratio to the longer run,
  SPY drawdown, TLT and USO 21 day return) and the sector flags. It is a
  pairwise ranker with one query group per date.
- A conductor combines them: score = confidence times the within-sector
  z-score of the expert, plus lambda times the per-date z-score of the
  supervisor. Confidence halves an expert's weight while its within-sector
  IC over the previous 63 dates is negative. Only forward windows that had
  already closed are counted, so there is no lookahead. This is the "dial
  down the tech model during a tech crash" mechanism.

Why no reinforcement learning or deep RankNet router. The rankings do not
move the market and every sector's return is observed every day, so there is
nothing to explore and no delayed consequence; an RL objective reduces to the
supervised one used here. There are also only about 170 non-overlapping
10-day periods in eight years, far too few to train an RL policy or a
neural router without overfitting.

Code. moe.py holds the design (features, conductor, save/load), moe_ranker.py
trains and evaluates it, metrics.py holds the vectorized IC and spread
shared by both training scripts. Models are saved in models/.
live_predictions.py --arch moe writes predictions_v6_*.csv. The monitor_
models in models/ are the train+val refit and are used only to measure
live expert confidence on dates they never saw.

Results, same cached data and splits as v5.
- Early stopping on validation AUC: shared expert 80 rounds, supervisor 14
  rounds. Per-sector experts stopped at 1 to 23 rounds for seven of ten
  sectors, which means too little data per sector to learn much alone.
- Validation within-sector IC: per-sector experts +0.043, shared +0.037.
- Supervisor sector-level IC (ordering 10 sectors per day): +0.106 on
  validation, +0.055 on test.
- Conductor grid on validation (2 expert kinds x confidence on/off x 6
  lambdas). Best: shared expert, confidence on, lambda 1.0, val IC +0.073
  against +0.058 for v5 on the same dates. Confidence on or off changed val
  IC by under 0.001 at every setting, which is noise.
- Held-out test, scored once: IC +0.033 (v5 +0.044), within-sector IC +0.049
  (same as v5, as expected since the expert is the same model), top minus
  bottom quintile +0.94 percent per 10 days (v5 +1.65 percent). Confidence
  dialled down 33 percent of (date, sector) rows.

Decision. v6 won validation but lost the one-time test, and the
supervisor's skill halved from validation to test. The validation period
was used twice for v6 (to early-stop the supervisor and to pick lambda from
24 settings), so its validation number is the most optimistic of any
version. v5 stays in production (config.MODEL_ARCH = 'single'). v6 runs in
shadow so it can earn or lose its place on live data.

A caution that applies to every comparison in this file. Daily IC has a
standard deviation near 0.2, and 10-day forward windows overlap, so a
one-year period holds only about 25 independent observations. The standard
error of a period's mean IC is roughly 0.04. Every gap between versions
recorded here, including v6 versus v5, is smaller than that. Live
validation over many windows is the only way to separate them.


VALIDATION SCHEDULE

predictions_v3_20260703.csv was generated from the July 2, 2026 close (July 3
was a market holiday). Its 10-trading-day horizon completes at the close on
Friday, July 17, 2026. To validate, on or after Saturday, July 18: uncomment
the predictions_v3_20260703.csv line in validate_predictions.py and run it.
It will produce validation_results_v3_20260703.csv with the same statistics
used in validations.md. predictions_v2_20260703.csv covers the identical
window and can be validated the same day for a v2-versus-v3 comparison.

predictions_v4_20260930.csv, predictions_v5_20260930.csv and
predictions_v6_20260930.csv were all generated after the September 30, 2026
close. Their horizon completes at the close on Wednesday, October 14, 2026.
On or after October 15, uncomment their lines in validate_predictions.py and
run it for a three-way comparison of both the rank IC and the within-sector
rank IC. To keep v6 collecting evidence, run live_predictions.py --arch moe
every time the default live_predictions.py is run.
What to expect if the model is working: rank IC above zero and the top
quintile beating the bottom quintile. Any single 10-day window is noisy, so
judge the model on several validation cycles, not one.


FILE MAP

- config.py: universe with sector labels, horizon, history length, purge gap,
  sector-specific features and the interaction constraint mode
- feature_engineering.py: feature construction, per-date normalization, target
- stock_ranker.py: trains, evaluates honestly, saves the production model
- moe_ranker.py, moe.py: v6 gated sector experts (shadow), saved to models/
- metrics.py: vectorized IC and quintile spread used by both trainers
- live_predictions.py: scores the universe today, writes predictions_v5_*.csv
  with sector ranks (or predictions_v6_*.csv with --arch moe)
- validate_predictions.py: scores any predictions file against realized prices
- validations.md: validation of the June 8 v1 predictions
- validation_results_*.csv: per-stock output of validate_predictions.py
