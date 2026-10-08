# Frozen model, costs disabled

At Simon's request, transaction costs are temporarily disabled. Borrow fees remain zero. The model, forecasts, dataset, capital allocation and rank-based holding rules are unchanged. This is an exploratory revision of the previously viewed 2024–2025 test.

## Measured results without costs

| Metric | Selected XGBoost Ranker strategy |
|---|---:|
| Initial capital, 2024-01-02 | $1,000,000 |
| Ending equity, 2025-12-31 | $1,081,594.78 |
| Cumulative return | +8.16% |
| Annualized Sharpe | 0.65 |
| Maximum drawdown | -6.98% |
| Mean daily rank IC | 0.0114 |
| Resolved return sessions | 501 / 501 |
| Transaction and borrow fees | $0 |
| Mean daily traded notional / prior equity | 73.59% |

Gross and net ledgers now coincide because both transaction and borrow fees are zero. The positive return is a result without trading costs. The final book holds 31 long and 24 short names, marked at the last close without forced liquidation.

## Rules and comparisons

The same saved ranker and complete forecast table drive next-session OHLC4 purchases in the top 20 and short additions in the bottom 20. Positions stay until prior-close ranks leave the top/bottom 100. Each signal close allocates available cash 50/50 between longs and short entry collateral; short proceeds remain reserved. Exit releases can fund orders at the fill-day close for the following session. All 501 return sessions resolve without missing selected-model fills, funding shortfalls or insolvency.

`leaderboard.csv` retains every frozen model and the reversal/momentum signals. XGBoost regression, CatBoost and momentum remain unresolved because they hold FISV when its adjusted-close quote is missing on 2025-11-12. Their portfolio metrics are unavailable; forecast IC remains separately measurable. These comparisons do not change the selected model.

## Experiment history and reproduction

The [previous 10 bps holding strategy](archive/hold-v2-10bps/RESULTS.md) ended at $748,075, with net return −25.19%. The [first intraday experiment](archive/intraday-v1/RESULTS.md) is also preserved. Current experiment ID: `f9068e97eaac`. Data, model, forecast and source hashes are recorded in `summary.json`.

Run `.venv/bin/python scripts/run_showcase.py` to verify and reuse the saved report. `--rebacktest` recomputes execution without model fitting. The dashboard is `index.html`; the notebook is `notebooks/SimonResearch_Showcase.ipynb`. Costs can later be restored through `config/rank_hold.json`.

See the [execution and accounting rules](../../docs/rank-hold-backtest.md). OHLC4 remains a hypothetical fill; synthetic adjusted units approximate corporate actions, and current-constituent/Yahoo data retains survivorship and revision bias. The eventual paper still needs stronger data and an independently reserved confirmation period.
