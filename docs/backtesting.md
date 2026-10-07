# Our shared backtest plan

Everyone builds a model. The leader keeps one dataset and one evaluator for fair comparisons. The evaluator still needs to be built.

```text
Shared data → your model → prediction CSV → shared evaluator → results
```

## 1. Use the same inputs

Use the notebook's nine features to predict next-day adjusted-close return. Train through 2022, choose model settings with 2023, and test once on 2024–2025. All stocks on a date stay in the same split.

Fit scaling and other learned data steps on training data only. Future returns and ranks aren't features. [Why this matters](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

Train once and keep the model fixed during testing. We can add retraining later.

Each member submits one CSV:

| Date | Ticker | prediction |
|---|---|---|
| 2025-01-02 | AAPL | 0.0012 |

`Date` is the completed trading day used for the prediction. `prediction` is next-day return as a decimal: `0.0012` means `0.12%`.

Submit one finite numeric prediction per eligible date and ticker, with no duplicates. The evaluator checks coverage so nobody can skip hard stocks or days. Keep rows without future targets for predictions; drop them for fitting and scoring.

## 2. Check the rankings

Compare predicted ranks with realized next-day return ranks using Spearman correlation on each test date, then average the scores. Positive scores mean higher predictions tend to lead to higher returns. [pandas calculation](https://pandas.pydata.org/docs/reference/api/pandas.Series.corr.html)

Report scored dates and undefined scores separately. Identical predictions, for example, give no ranking to compare. Don't count that as zero or success. Ranking quality isn't trading profit.

## 3. Give every model the same trading rules

- After the close on day `t`, sort predictions from highest to lowest. Break ties by ticker.
- Buy the top 10, with 10% of the portfolio planned for each. No short positions. If fewer qualify, keep spare slots in cash.
- Buy at the next market session's open and sell at its close. Hold cash overnight and repeat each day.
- Charge **0.10% on each buy and each sell** — 10 basis points each way. Report returns before and after costs, charging both sides of every executed trade.

The target includes the overnight move from close to close; our trades run from the next open to close. Calculate profit from those actual Open and Close prices, never from `target`. Full-day features only support trading afterward.

Use one market calendar: a stock's next available quote might skip a session. Include only holding periods covered by the snapshot. Report final dates left out because the holding period ends after the data does.

Decide which stocks qualify using only information available on day `t`, without requiring future labels. Rank before checking tomorrow's prices. A selected stock with no valid opening price keeps its slot in cash; don't replace it. A bought stock with no closing price makes the result unresolved. Flag it; don't drop the trade or assume zero return.

Compare each model with an equal-weight portfolio of all eligible stocks, using the same trading times and costs.

## 4. Keep the code small

Use separate model notebooks. Add one function in `src/backtests/backtest.py` that takes predictions and saved quotes, returning daily portfolio returns and a results table.

Save predictions and results in `reports/`. Keep shared generated data in `data/processed/`.

Compare these numbers:

- **Mean daily Spearman score:** how well the model ranks stocks.
- **Cumulative net return:** the total gain after costs.
- **Annualized daily Sharpe ratio:** return compared with daily ups and downs, scaled to a year.
- **Maximum drawdown:** the biggest drop from a previous portfolio high.

Share the exact evaluation dates, trading rules, costs, and benchmark with the results.

The downloaded stock list can favor survivors when studying past years — survivorship bias. Quote and feature checks keep this exercise usable; they don't prove past S&P 500 membership or real-world tradability.
