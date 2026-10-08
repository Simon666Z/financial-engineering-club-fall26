# Frozen model, revised holding strategy

The same validation-selected XGBoost Ranker and complete saved forecasts are retained. The data still comes from `Data.ipynb`: 499 usable securities, nine raw features and nine observable same-day percentile transforms; training through 2022, validation in 2023, and 2024–2025 evaluation. The new trading rule was requested after the original test had been viewed, so this is an exploratory strategy revision.

## Measured results

| Metric | Selected model strategy |
|---|---:|
| Initial capital, 2024-01-02 | $1,000,000 |
| Ending gross equity, 2025-12-31 | $1,081,594.78 |
| Ending net equity, 2025-12-31 | $748,075.29 |
| Gross cumulative return | +8.16% |
| Net cumulative return | -25.19% |
| Annualized net Sharpe | -2.25 |
| Maximum net drawdown | -26.54% |
| Mean daily rank IC | 0.0114 |
| Resolved return sessions | 501 / 501 |
| Transaction fees in net book | $319,859.81 |
| Mean daily traded notional / prior NAV | 73.55% |

The strategy adds longs from the top 20 and shorts from the bottom 20 using ranks known at the prior close. Longs exit after leaving the top 100; shorts after leaving the bottom 100. Both fills occur at the next session's OHLC4 proxy. The first fills are January 3, 2024. Available capital at each signal close is split 50/50 between long purchases and short entry collateral; short proceeds and collateral remain reserved. Existing qualifying positions are kept, so position counts can exceed 20 on either side. Exit releases can fund orders at that day's close for the following session.

The final net book holds 31 long and 24 short names. Ending equity includes their marked value; there is no invented terminal liquidation or closing commission. There were no missing fills, funding shortfalls or insolvency events in either selected-model book. Gross and net are independently funded simulations, so their difference includes both fees and the resulting sizing/compounding differences.

Holding the positions longer does not establish profitability. The gross gain remains overwhelmed by turnover and the stated 10 bps commission per trade side. Borrow fees are zero and every selected security is assumed shortable, so realistic stock-loan frictions would add further constraints.

## Comparisons and experiment history

`leaderboard.csv` records every frozen model and the reversal/momentum signals under the same rule. XGBoost regression, CatBoost and momentum cannot be valued completely: they hold FISV when its adjusted-close quote is missing on 2025-11-12. Their portfolio metrics are explicitly unresolved, not flat-return estimates. Their forecast IC remains measurable. The selected ranker is unchanged by these comparisons.

The original intraday experiment is preserved in [archive/intraday-v1/RESULTS.md](archive/intraday-v1/RESULTS.md), with gross +10.50% and net −59.43%. Its negative result is retained alongside this revision. Model, forecast, raw and prepared data hashes, strategy settings and source hashes are recorded in `summary.json`. Current experiment ID: `8f1bcf189390`.

## Reproduce

Run `.venv/bin/python scripts/run_showcase.py` to verify and reuse the cached report, or add `--rebacktest` to recompute execution with the frozen model. Open `notebooks/SimonResearch_Showcase.ipynb` for the recorded experiment and `index.html` for the simple model/strategy dashboard. `--retrain` explicitly fits again; `--refresh-data` changes and rebuilds the snapshot.

The exact [execution and accounting rules](../../docs/rank-hold-backtest.md) distinguish causal order decisions from dataset quality. OHLC4 is hypothetical, adjusted units provide synthetic corporate-action accounting rather than actual historical shares/dividend cashflows, and current-constituent/Yahoo data retains survivorship and revision bias. Paper research requires stronger data and an independently reserved confirmation period.
