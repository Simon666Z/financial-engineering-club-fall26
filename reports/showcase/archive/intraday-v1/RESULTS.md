# First frozen showcase experiment

This demonstration uses the `Data.ipynb` universe/date scope and nine original features, plus target-free same-day percentile transforms. The download produced 1,183,015 eligible rows for 499 securities. Model fitting uses 2016–2022; 2023 controls early stopping, blend selection, and return calibration; 2024–2025 measures the frozen choice.

Four fixed configurations were trained: ElasticNet, XGBoost rank-target regression, XGBoost LambdaMART ranking, and CatBoost rank-target regression. Twelve fixed blend candidates included single models. Validation selected XGBoost ranking alone; its best boosting iteration was 68 (zero-indexed). The other configurations and all blend trials remain in `summary.json`.

## Recorded test results

| Metric | Selected model |
|---|---:|
| Mean daily rank IC | 0.0114 |
| Gross cumulative return | +10.50% |
| Net cumulative return | −59.43% |
| Annualized net Sharpe | −2.29 |
| Maximum net drawdown | −60.92% |
| Resolved holding sessions | 501 / 501 |

The strategy ranks after close, allocates equal funded capital slots to the top ten, enters at the next session's open, and exits at that session's close. Cash is held overnight. The cost assumption is 10 bps on each buy and sell. Entry commissions are funded within the allocation. No missing-price trade was silently marked as a flat return.

The positive gross result does not support profitability under this cost assumption. Rank IC is small; the simple reversal baseline and several other models have better test rank IC and gross performance. They are reported as comparisons, and were not substituted for the model selected on validation. `leaderboard.csv` preserves every result. The next-day forecast label includes overnight returns, while this execution rule earns only intraday returns; these are distinct outcomes.

## Reproduce and present

Run `.venv/bin/python scripts/run_showcase.py` to reuse the completed frozen experiment, or open `notebooks/SimonResearch_Showcase.ipynb`. The dashboard is `index.html`; it shows the model scheme and strategy equity before and after costs together. Data snapshots and fitted models are retained locally and excluded from Git. Their hashes, package versions, settings, stopping iterations, calibrated return units, and split boundaries are recorded in `summary.json`.

A deliberate new fit uses `--retrain`; a new download uses `--refresh-data`. Test results have now been viewed. Any later development on 2024–2025 must be labeled exploratory, with a new independent confirmation sample reserved for the eventual paper.

## Implication for the publication goal

This milestone delivers the technical pipeline and an honest negative trading result. It does not establish a profitable strategy or a research contribution. The evidence suggests future work should examine target/execution alignment, turnover, and whether model complexity adds value beyond simple signals. Those questions need literature review and stronger data before becoming a paper thesis. Current-constituent survivorship bias, revised Yahoo history, simplified costs, and missing market-impact modeling remain material limitations.
