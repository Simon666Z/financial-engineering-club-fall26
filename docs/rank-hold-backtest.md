# Persistent long-only backtest

The current strategy uses the saved expanded-feature models and forecasts. It begins with **$1 million on 2024-01-02**, with first fills on 2024-01-03. Every individual model and the validation-selected blend receive their own independently accounted portfolio.

## Daily sequence

1. At the completed close, rank every eligible stock from its known price/volume features.
2. Allocate all already-available free cash equally to the **top 20** for next-session entries. Existing selected holdings may receive additions.
3. Retain a long while its descending rank is **100 or better**. Rank 101 or no eligible rank schedules an exit for the following session. Retained stocks can make the portfolio exceed 20 names; this is not a full daily rebalance.
4. Execute yesterday's orders at the current session's adjusted OHLC4, mark positions at adjusted close, and then plan the next session's orders.

Cash released by an exit on Wednesday becomes available at Wednesday's close for Thursday's entries. Wednesday's entries cannot use those unknown proceeds when planned on Tuesday. Missing entries retain their allocation as cash; missing exits retain the holding for another attempt. Missing held marks invalidate evaluation rather than receiving a stale price or zero return.

## Prices and inventory

```text
OHLC4 = (Open + High + Low + Close) / 4
AdjustmentFactor = Adj Close / Close
ExecutionAverage = AdjustmentFactor * OHLC4
MarkClose = Adj Close
Equity = free cash + marked long assets
```

OHLC4 is a simulated full-day fill known after that day's close. Orders use only the prior completed session. The adjustment is consistent with [yfinance's auto_adjust scaling](https://github.com/ranaroussi/yfinance/blob/main/yfinance/utils.py). Quantities are synthetic total-return units; they embed price adjustments and do not add separate dividend cash or split-share credits.

The book uses no shorts or borrowing. Initial gross exposure is funded by its cash, and holdings can concentrate through repeated additions; there is no position cap or stop. Final positions are marked without a forced terminal sale. Cash earns zero. Transaction costs currently remain **0 bps** at Simon's request. Spread, impact, volume participation and actual dividend cash timing are not modeled.

## Inspect every model

```bash
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebacktest
```

This replays all portfolios with the saved model and forecasts. `--retrain` is a separate explicit fit; `--rebuild-features` requires it. Settings live in [config/rank_hold.json](../config/rank_hold.json).

The [dashboard](../reports/alpha/index.html) and [notebook](../notebooks/SimonResearch_Showcase.ipynb) always show individual model performance and the blend. `leaderboard.csv` has every recorded result; `model_daily.csv` preserves each resolved NAV path. The primary `daily.csv`, `trades.parquet` and `positions.parquet` audit the selected blend. Invalid or unresolved models remain visible with unavailable metrics.

## Earlier strategy and limits

The prior long/short alpha experiment is preserved in [reports/alpha/archive/long-short-v1](../reports/alpha/archive/long-short-v1/RESULTS.md). Its engine remains supported: free capital was split 50/50, short sale proceeds and equal entry collateral were segregated, and short exits used bottom 100 ranks.

The universe is current at download rather than historical membership, and Yahoo quotes can be revised. All 2024–2025 results are exploratory after repeated strategy development. Long-only gains include market exposure. These historical assumptions do not establish live trading profit.
