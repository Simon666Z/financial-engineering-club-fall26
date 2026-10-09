# SimonResearch: demonstration first, publication goal retained

Simon is the group leader and wants a compelling working model before selecting the eventual paper thesis. The personal research branch is `SimonResearch`; the original FE Club chat and checkout focus on `main`. The ultimate objective is a submitted, publishable research paper. A strong demo is the first milestone, not evidence that the publication objective is already achieved.

## Current scope: the notebook data

Build and improve the model using the universe and price scope in `notebooks/Data.ipynb`. Keep the original nine features as a saved baseline:

- `ret_1d`, `mom_5d`, `mom_20d`, `mom_60d`
- `vol_5d`, `vol_20d`, `volume_ratio_20d`
- `range_pct`, `intraday_ret`

Use the chronological training period through 2022, 2023 validation, and 2024–2025 test. Fit and select models using the earlier periods, record all model choices, and distinguish next-day learning targets from executable portfolio returns. Record the configured costs, drawdowns, simple baselines and unresolved execution observations; keep the dashboard simple. The current priority is measured model and strategy performance. Paper planning comes after the working model. Add the requested price/volume features without changing the stock universe or date range.

The run command is `python scripts/run_showcase.py` in the research environment. `notebooks/SimonResearch_Showcase.ipynb` explains and displays the experiment. `reports/alpha/index.html` is the current local presentation artifact; its `summary.json` and accompanying CSVs are the evidence. The original experiment remains in `reports/showcase/`.

## What the demo can establish

It can show that the pipeline runs, models learn measurable relationships, the ensemble selection is reproducible, and forecast quality and portfolio behavior can be compared under stated assumptions. A disappointing result remains useful evidence. Performance should be described from the recorded output rather than promised in advance.

The data uses current constituents rather than historical membership. Yahoo-derived price snapshots can change, costs are a simplified assumption, and historical tests are not live execution. Save data provenance and hashes. If the 2024–2025 window guides later development, label it exploratory instead of treating it as an untouched research confirmation window.

## After the demo: develop the research contribution

1. **Understand model behavior.** Run ablations and comparisons to learn which signals, nonlinearities or trading choices explain a result. Possible starting questions include whether temporal complexity adds information beyond familiar features, and whether uncertainty improves trading decisions beyond reduced turnover or risk.
2. **Build a literature map.** Read papers supplied by Simon or located through primary sources. For each, record the research claim, dataset, chronology, baselines, assumptions, evaluation, limitations and relevance. Verify the novelty of a proposed question before making a contribution claim.
3. **Upgrade data as the question requires.** Consider historical universe membership, information availability dates, survivorship treatment, corporate actions, additional horizons and suitable execution data. The future paper's scope may differ from this teaching dataset.
4. **Specify a falsifiable hypothesis and confirmation design.** Keep exploratory results distinct from later confirmation. Reserve a new evaluation before final tuning, use realistic costs and robust baselines, and assess statistical uncertainty and multiple experimentation.
5. **Write alongside the evidence.** Maintain methods, experiment logs and reproducible figures. Draft the paper around the verified contribution; seek faculty feedback and choose an appropriate venue after the evidence supports the question.

Proceed through the demonstration now without adding approval gates. Publication timing, venue, faculty collaboration and data expansion can be settled when they become relevant.

## Reproducibility record

Keep the original `Data.ipynb` unchanged. Research artifacts should retain the experiment ID, generation timestamp, data snapshot SHA-256, train/validation/test spans, fitted parameters, selected ensemble weights, complete model leaderboard and explicit limitations. Record failures and alternative experiments as well as favorable results. Never merge private research changes into the group branch unless Simon requests it.

## First measured result — 7 October 2026

The first frozen run selected XGBoost ranking alone on 2023 validation. On 501 fully resolved test holding sessions, it recorded rank IC 0.0114, gross cumulative return +10.50%, net cumulative return −59.43% at 10 bps per side, and net Sharpe −2.29. Several components and the simple reversal baseline performed better on the test. The choice was kept frozen rather than replaced after seeing that comparison.

The pipeline works; a profitable strategy has not been established. Preserve this result in the experiment history. The next exploratory iteration can examine target/execution alignment, turnover reduction and strong simple baselines. A future publication must use a separately reserved confirmation sample, since this test has now been inspected.

## Revised holding strategy — 7 October 2026

At Simon's request, the same frozen model and forecasts now drive a $1 million long/short strategy: next-session OHLC4 fills for top/bottom 20 additions, exits after prior-close ranks leave top/bottom 100, and daily allocation of available cash with short proceeds reserved. The selected book resolves all 501 return sessions, ending at $748,075 after 10 bps each side: gross +8.16%, net -25.19%, Sharpe -2.25. Turnover still overwhelms the gross gain. The original result is archived; no model was selected using the revised test profit.

This is exploratory execution development, not independent confirmation. Before a paper, determine whether rank persistence and holding rules contain reproducible information beyond turnover changes and simple reversal. Strengthen execution/corporate-action/borrow data and reserve an untouched evaluation window.

## Current run: costs temporarily disabled

At Simon's request, transaction costs are set to zero; borrow fees remain zero. The same frozen model and holding rules end at $1,081,595, with cumulative return +8.16%, Sharpe 0.65 and drawdown -6.98%. This describes returns without trading costs. The earlier 10 bps result is preserved in `reports/showcase/archive/hold-v2-10bps/`; restore realistic costs before assessing economic value for the paper.


## Expanded features — 8 October 2026

Simon requested the feature experiment before further paper work. The expanded schema keeps the original nine inputs and adds 15 features across skip-month momentum, proximity to the annual high, unusual dollar volume, volume-conditioned reversal, MAX and overnight/intraday decomposition. Each raw input also has an observable same-date percentile, giving 48 model inputs. [Exact formulas](alpha-features.md) are saved with the data manifest.

The original raw snapshot, next-day target, model configurations and zero-cost holding strategy stay fixed. Complete one-year windows reduce the available training history; 2023 still selects stopping iterations and weights. A saved original-model comparison uses the same eligible test stocks. The current measured outcome is recorded in [the new results](../reports/alpha/RESULTS.md).
