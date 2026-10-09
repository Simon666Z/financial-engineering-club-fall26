# Expanded-feature model — measured run

The new validation-selected model has a positive **+9.77% cumulative return with costs excluded**, ending at **$1,097,653.29** from $1 million. Its next-day ranking quality improves, but its Sharpe and drawdown are worse than the saved original model on the same eligible test stocks. This is a completed feature experiment, not yet a consistently stronger trading strategy.

## Selected model and data

- **Inputs:** original nine plus 15 research features = 24 raw inputs and 24 observable daily percentile transforms (48 total). [Formulas and sources](../../docs/alpha-features.md).
- **Selected on 2023:** 25% ElasticNet, 25% XGBoost regression, 25% XGBoost LambdaMART ranking, 25% CatBoost. Twelve fixed blend candidates; validation mean daily rank IC selects the blend. No test result changes this choice.
- **Stopping iterations:** XGBoost regression 72, ranking 39, CatBoost 58, using zero-based best-iteration indices (predictions use 73, 40 and 59 rounds respectively).
- **Target:** next-session adjusted-close return, learned as daily return ranks (regressors) or ten return relevance grades (ranker). The separately exported decimal return estimate is calibrated on 2023.
- **Raw data:** unchanged saved notebook prices/constituents. Snapshot SHA-256 `4aa57a8e505bd98a121b59ec0b615ef74ec99183dfdd11e4bf111ee877b36d5c`. No new price download.

| Split | Signal dates | Rows | Rows with known labels |
|---|---|---:|---:|
| Train | 2017-01-03 → 2022-12-29 | 712,325 | 712,313 |
| Validation | 2023-01-03 → 2023-12-28 | 122,255 | 122,255 |
| Test | 2024-01-02 → 2025-12-31 | 248,377 | 247,879 |

The complete one-year windows delay first training signals until 2017-01-03 and reduce the usable history. Boundary-crossing training/validation labels are purged. Missing future returns do not remove observable signal rows; the final signals remain in forecasts even when no next-session label/fill exists. The expanded table covers 498 securities over the full period, with changing daily eligibility.

## Matched test comparison

Both rows below use the expanded experiment's same eligible test cross-sections and the same strategy. The original saved model was trained on its longer original history, so this compares two models; it does not isolate the causal effect of adding features.

| Metric | Original model, matched stocks | New selected blend |
|---|---:|---:|
| Ending equity | $1,086,157.50 | $1,097,653.29 |
| Cumulative return, no costs | +8.62% | +9.77% |
| Annualized Sharpe | 0.686 | 0.465 |
| Maximum drawdown | -6.50% | -18.83% |
| Mean daily next-day rank IC | 0.01138 | 0.02829 |
| Mean daily traded notional / prior equity | 73.65% | 55.80% |

The earlier original-universe result remains [archived in its existing report](../showcase/RESULTS.md): +8.16%, Sharpe 0.650, drawdown −6.98%. Its eligible test population differs slightly from the matched comparison above.

## Trading result and full comparison

The $1 million book starts 2024-01-02, with first fills 2024-01-03. Closing scores schedule next-session top 20 long/bottom 20 short additions at adjusted OHLC4. Longs remain while top 100 and shorts while bottom 100. Free cash is allocated 50/50; entry short collateral and sale proceeds are reserved. Exit releases fund orders planned after that execution day's close. [Accounting rules](../../docs/rank-hold-backtest.md).

The selected book resolves all 501 return sessions, with zero missing entry/exit fills, no funding shortfall or insolvency, and identical gross/net paths. Final equity includes open positions, without forced terminal liquidation. Costs and borrow fees are both zero.

| Model / baseline | Rank IC | Return, no costs | Sharpe | Drawdown | Portfolio status |
|---|---:|---:|---:|---:|---|
| ensemble | 0.02829 | +9.77% | 0.465 | -18.83% | resolved |
| elastic_net | 0.02932 | — | — | — | unresolved |
| xgb_regression | 0.02560 | -1.89% | 0.008 | -22.94% | resolved |
| xgb_ranker | 0.02561 | +13.54% | 0.813 | -9.09% | resolved |
| catboost | 0.02572 | +7.92% | 0.354 | -20.15% | resolved |
| reversal | 0.01644 | +8.10% | 0.587 | -8.02% | resolved |
| momentum | -0.00735 | — | — | — | unresolved |
| notebook_baseline | 0.01138 | +8.62% | 0.686 | -6.50% | resolved |

The largest selected-book daily loss was −7.84% on 2024-11-15, dominated by a retained BE short. The saved closing price rose from $13.28 to $21.14 (+59.19%) with an unchanged adjustment factor. A further BE short loss dominated 2024-11-22. Its accumulated short exposure reached 39.26% of NAV on 2024-12-02; BE explains about 79% of the peak-to-trough dollar loss. Repeated additions and retained holdings can concentrate exposure because this strategy has no position cap or stop. These observations come from the saved quotes and position ledger; no external event explanation is assumed.

The expanded XGBoost Ranker alone has a better observed test strategy result than the selected blend. It is reported as an exploratory comparison; the validation-selected blend remains the primary result. ElasticNet and momentum hold FISV when its 2025-11-12 closing mark is missing; their full-period portfolio results remain unresolved. No stale price or zero return is substituted, and the ticker was not excluded using future quote availability.

## Inspect and reproduce

```bash
# Verify saved artifacts and reuse this completed run
.venv/bin/python scripts/run_showcase.py --feature-set alpha
# Recompute this strategy using its saved model and forecasts
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebacktest
# Explicit new feature rebuild and fit, using saved raw data
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebuild-features --retrain
```

Open [the dashboard](index.html) or [the executed notebook](../../notebooks/SimonResearch_Showcase.ipynb). `summary.json` records hashes, split dates, protocol and every comparison; `training.json` retains candidate parameters and all validation trials. `daily.csv` contains the equity observations. Complete forecasts, trades, positions and fitted models stay local as ignored large artifacts.

The universe uses current-at-download membership, so historical results have survivorship and membership bias. OHLC4 is a hypothetical fill proxy; the prices are synthetic adjusted total-return units. The 2024–2025 window has already informed exploratory development. These limitations and the zero-cost assumption remain part of the measured result.
