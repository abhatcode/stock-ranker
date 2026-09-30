This project ranks a list of about 165 stocks from best to worst expected performance over the next two weeks. It does not predict prices. It only orders stocks against each other, using an XGBoost model trained on price and volume history.

Why rank instead of predict a price or a return. Predicting an exact return is close to impossible from price data alone, and the accuracy of an exact prediction is not what matters for picking stocks anyway. What matters is whether the stocks the model ranks highest actually do better than the stocks it ranks lowest. That is a ranking problem, so the model is trained and judged as one.

How it works, in order.

feature_engineering.py turns each stock's daily prices and volume into about 30 numbers per day: recent returns at several lengths, momentum, volatility, how far the price is from its recent high, trading volume trends, and a few others. Every number is scale-free, meaning it is a ratio or a percentile, never a raw price or dollar amount, so a $5 stock and a $500 stock are compared fairly. Each number is also compared only against the same day's numbers for every other stock, so the model is always judging "better or worse than the rest of the market today," not absolute levels that drift over months and years.

stock_ranker.py trains the model. It splits history into three time periods: train, validation, and test, with a gap between them so a return that spans the boundary cannot leak from one period into another. It trains on the first period, tunes on the second, and only checks final performance on the third, once, so the reported numbers are honest and not the result of picking whatever worked best on the test data.

live_predictions.py downloads the latest prices and runs the trained model on today's data, producing a ranked list saved as a CSV file.

validate_predictions.py waits for the prediction's two-week window to pass, then checks what actually happened and reports whether the ranking was any good.

Two problems we ran into and fixed.

The first models mixed up stocks from different industries too freely. A feature that matters for one kind of company, such as a bank's sensitivity to interest rates, would end up influencing the ranking of a software company where it has no meaning, adding noise instead of signal. We fixed this two ways: by telling XGBoost which features are allowed to combine with which sector (see INTERACTION_MODE in config.py), and by adding a couple of features that only exist for the sectors they are relevant to, left blank everywhere else.

The second problem was that ranking every stock against the entire list at once taught the model to sort well overall, but it never specifically learned what makes one tech stock beat another tech stock, since that comparison was always mixed in with unrelated sectors. We tried a version that ranks stocks only against others in their own sector (TARGET_MODE in config.py), and it did noticeably better within a sector while giving up a little on the cross-sector picture. That version, called v5, is what currently runs live.

We also tried a more elaborate fix: separate expert models for each sector, plus a second model whose only job is to rank the ten sectors against each other, combined into one final score. This is in moe_ranker.py and moe.py, referred to as v6. It looked better than v5 on the tuning data, but did worse on the final untouched test period, so it is not live yet. It still runs every day alongside v5 so we can build up more evidence before deciding whether to switch.

Read spec_decisions.md for the full history of what we tried, what worked, what did not, and why, in the order we tried it. It is kept in plain text on purpose, so it is easy to read straight through without needing anything to render it.

Running it.

Install the packages in requirements.txt. Then:

python stock_ranker.py trains the v5 model and saves trained_model.json.
python moe_ranker.py trains the v6 models and saves them in the models folder.
python live_predictions.py ranks today's stocks with v5 and saves a predictions file. Add --arch moe to do the same with v6.
python validate_predictions.py checks any past predictions file whose two-week window has already passed, listed near the top of that file.

A GPU is used if one is available but is not required; training just takes longer on a CPU.

What the numbers in this project mean. IC, short for information coefficient, is how well the model's ranking matches what actually happened that day, from minus one (completely backwards) to plus one (perfect). Anything consistently above zero across many days means the ranking has real skill. Quintile spread is the average difference in return between the top fifth of the ranked list and the bottom fifth. These numbers are small in this project, usually a few percent, because price and volume data alone carry a weak signal. That is expected and is not a sign of a bug; a very large number here would be a sign that something is leaking information it should not have.
