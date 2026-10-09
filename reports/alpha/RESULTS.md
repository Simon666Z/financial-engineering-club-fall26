# Long-only model comparison

The validation-selected four-model blend turns $1 million into **$1,709,546.02** in 2024–2025: **+70.95%**, annualized Sharpe **1.835**, and maximum drawdown **-8.79%**, with costs excluded. Every saved model is evaluated separately using identical prices, eligible stocks and trading rules. The original core models and forecast table remain frozen. The completed TabPFN candidate adds a separate comparison using the same data and execution rules.

## Trading scheme

Start with $1 million at the close of 2024-01-02. Budget **all already-available free cash** equally among the top 20 ranked stocks. The first fills occur 2024-01-03 at that session's synthetic adjusted OHLC4. Retain existing holdings while ranked in the top 100; a rank above 100 or loss of eligibility schedules an exit for the following session. Additions can increase existing holdings, and retained stocks can make the book exceed 20 names.

Exit proceeds become available at that execution day's close for the next session's orders. There are no shorts, leverage or short collateral. Final holdings are marked at the final observed close; no forced liquidation is invented. The base scenario has zero fees; the separate return column charges 10 bps per buy and sell. [Full accounting protocol](../../docs/rank-hold-backtest.md).

## Every model and the blend

| Model | Return before costs | Return after costs (10 bps/side) | Sharpe before costs | Maximum drawdown before costs | Ending equity before costs | Status |
|---|---:|---:|---:|---:|---:|---|
| Blend (25% each) | +70.95% | +23.03% | 1.835 | -8.79% | $1,709,546.02 | resolved |
| Elastic Net | +73.76% | +36.40% | 1.680 | -12.27% | $1,737,623.24 | resolved |
| XGBoost regression | +56.42% | +14.28% | 1.280 | -18.63% | $1,564,215.01 | resolved |
| XGBoost Ranker | +66.23% | +15.19% | 1.877 | -11.55% | $1,662,283.70 | resolved |
| CatBoost | +71.38% | +23.46% | 1.646 | -12.81% | $1,713,805.49 | resolved |
| Reversal baseline | +50.19% | -2.00% | 1.526 | -8.13% | $1,501,903.34 | resolved |
| Momentum baseline | +181.68% | +164.77% | 1.898 | -26.67% | $2,816,810.27 | resolved |
| Original-feature baseline | +39.94% | -3.08% | 1.435 | -10.31% | $1,399,355.12 | resolved |
| TabPFN-3.5 | +252.51% | +223.78% | 1.392 | -39.40% | $3,525,086.84 | resolved |

The separate after-cost return uses **10 bps (0.10%) of every filled buy and sell**. Each model is replayed with its own fee-funded cash and holdings, so fees reduce later investable capital and compounding. All other table metrics and plotted curves retain the 0 bps base scenario. Open final holdings incur no invented liquidation fee. Costed curves are saved in `cost_model_daily.csv`; the blend's fee ledger and holdings are saved in `cost_trades.parquet` and `cost_positions.parquet`.

The four-model blend was selected on 2023 ranking accuracy and remains 25% each ElasticNet, XGBoost regression, XGBoost Ranker and CatBoost. The test comparison does not change those weights. The momentum baseline's observed return exceeds every original local model; profit alone does not establish that model complexity added value. Long-only profits include broad equity exposure and are not a market-neutral alpha estimate.

TabPFN has the highest cumulative return but a larger drawdown and lower Sharpe than the blend. This does not make it the best strategy for every risk objective.

The original-feature baseline is the saved original model applied to the same expanded eligible test stocks. It has a different training history. All available resolved model curves are shown separately in the dashboard and `model_daily.csv`; the blend is highlighted. A missing or invalid model remains in the table with its status and no invented portfolio performance.

## Frozen experiment and checks

- 24 raw signals + 24 target-free same-day percentile transforms = 48 inputs. [Exact features](../../docs/alpha-features.md).
- Training signal dates 2017-01-03 to 2022-12-29, validation 2023-01-03 to 2023-12-28, test 2024-01-02 to 2025-12-31. Training/validation labels crossing their boundaries are purged.
- Original saved raw snapshot is unchanged. No price data or new fitted core model was fetched for this execution revision.
- The selected book resolves all 501 return sessions. Its short inventory and collateral stay zero, gross/net paths coincide, and entries/exits lag their signals by one observed session.
- The previous long/short result and its ledgers remain in [archive/long-short-v1](archive/long-short-v1/RESULTS.md).

The data's current-at-download membership has survivorship and membership bias. OHLC4 is hypothetical execution, and adjusted quotes simulate total-return units. This repeatedly viewed 2024–2025 window is exploratory. Fees, spread and market impact are excluded. These assumptions remain attached to all displayed performance.

## Reproduce

```bash
.venv/bin/python scripts/run_showcase.py --feature-set alpha
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebacktest
```

The first command checks hashes and reuses the complete run. The second replays execution using the same saved forecasts. Open [the dashboard](index.html), [the notebook](../../notebooks/SimonResearch_Showcase.ipynb), `summary.json`, `leaderboard.csv` and `model_daily.csv`. TabPFN-3.5 completed all 122,255 validation and 248,377 test forecasts using a fixed 10,000-row query protocol that passed the original causal gate. It produces +252.51% cumulative return with a 39.40% maximum drawdown and 1.392 Sharpe. Its context uses 20,000 training rows, compared with 712,313 labeled rows for the core models. It remains outside the frozen blend. Full inference consumed 390,000 tokens; all diagnostics and inference together consumed 480,000. [Protocol, evidence and budget](../../docs/tabpfn.md).
