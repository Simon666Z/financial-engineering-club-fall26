# A simple shared backtest

Each member should build a model, while the group leader maintains one dataset and one evaluation function. Start by comparing prediction rankings; then run every model through the same small portfolio simulation. This document proposes that function; it does not implement it.

```text
Frozen data → member's model → prediction CSV → shared evaluation → results
```

## 1. Everyone uses the same inputs

Use the notebook's nine features and next-day close-to-close return target. Keep the shared chronological training, validation, and test periods. All stocks on one date belong to the same period. Fit any scaling or other learned preprocessing only on training data; select model settings on validation data and evaluate the final model once on test data. [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

For the first version, train once and hold the model fixed during the test period. Rolling retraining can wait.

Each member exports one CSV containing:

| Date | Ticker | prediction |
|---|---|---|
| 2025-01-02 | AAPL | 0.0012 |

`Date` is the date whose completed trading-day features produced the prediction. `prediction` is predicted next-day return, expressed as a decimal. Submit one finite prediction per eligible date/ticker, without duplicates. The evaluator checks the shared expected coverage so members cannot quietly omit difficult stocks or days.

## 2. First compare ranking quality

For each test date, compute Spearman correlation between predictions and the notebook's realized next-day close-to-close returns; then average the daily correlations. Positive values indicate that higher predictions tend to identify higher subsequent returns. [pandas supports `corr(method="spearman")`](https://pandas.pydata.org/docs/reference/api/pandas.Series.corr.html).

Report the number of evaluated dates and undefined correlations separately. A day with constant predictions has undefined rank correlation; it is not a zero or a successful day. Ranking results are a prediction check, not a trading profit calculation.

## 3. One portfolio rule for every model

- After the close on date `t`, rank stocks by their predictions. Break ties by ticker.
- Choose the top 10, long only, with equal intended weights. If fewer than 10 are eligible, leave unused slots in cash.
- Buy at the next trading session's open and sell at that session's close. Hold cash overnight; repeat daily.
- Use the same fixed assumption of **10 basis points (0.10%) on each buy and each sell**. Report both gross and net results, charging both sides on each executed position.

The learning target remains close-to-close for simplicity. Its overnight component differs from the simulation's open-to-close holding period. Compute portfolio profit from the actual next-session Open and Close, never directly from `target`. Completed-day features cannot support buying earlier at that day's close.

Use one shared market-session calendar; a ticker's next available quote could skip a session. Evaluate only holding periods fully covered by the quote snapshot, and report excluded final horizon dates.

Determine eligibility using information available on signal date `t`, separately from future label availability. Rank before examining future prices. If a chosen stock lacks a valid next-session opening quote, keep its intended slot in cash; do not substitute another stock. If an entered position has no exit quote, flag the run for unresolved valuation instead of dropping the position or assigning zero return.

For comparison, run an equal-weight portfolio of the eligible universe using the same open-to-close timing and cost assumptions.

## 4. Keep the code and results small

Members work in their own model notebooks. Later, add one shared function in `src/backtests/backtest.py` that accepts predictions and the quote panel and returns daily portfolio returns plus a results table. Save prediction files and results under `reports/`; keep shared generated inputs under `data/processed/`.

Compare mean daily rank correlation, cumulative net return, annualized daily Sharpe ratio, and maximum drawdown. Publish the exact common dates, portfolio rules, costs, and benchmark beside the results.

The sample uses today's constituent list over past dates, so it has survivorship bias. Available quotes and features are an educational eligibility screen, not historical S&P 500 membership or proof of real-world tradability.
