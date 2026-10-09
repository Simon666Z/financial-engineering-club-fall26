# FE Club · Fall 2026

Pick a model, predict next-day stock returns, and rank the stocks. Everyone uses the same data so we can compare models fairly.

## Start here

Run [`notebooks/Data.ipynb`](notebooks/Data.ipynb) in order. It cleans prices, builds the original nine features, and splits the data by date:

- **Train:** through 2022 — teach your model.
- **Validation:** 2023 — choose your model settings.
- **Test:** 2024–2025 — check the final result.

From the cloned repo, install the packages and open the notebook:

```bash
python -m pip install jupyter -r requirements-data.txt
jupyter notebook notebooks/Data.ipynb
```

In Colab, run these first, then run the notebook cells in order:

```python
!git clone https://github.com/Simon666Z/financial-engineering-club-fall26.git
%cd financial-engineering-club-fall26
!pip install -r requirements-data.txt
```

## Try things on your own branch

Please create your own branch for anything that isn't going into our final project, like picking a model or doing trial runs with the data. Keep `main` for work we're including in the final project.

## Use the same data

Feel free to add your own features, but for now I
added some already.

Keep rows without a future return for predictions; drop them when training or scoring. Don't use `target_rank`, dates, or extra columns as features.

The downloaded list doesn't track past S&P 500 membership, so results can favor stocks that survived. The target is a learning label; trading profit needs its own price calculation. See the [shared backtest plan](docs/backtesting.md).

Run the data checks with:

```bash
python -m unittest discover -s tests -v
```

## Where things go

```text
.
├── config/                 # Settings and paths
├── data/
│   ├── external/           # Outside reference data
│   ├── interim/            # Data between cleaning steps
│   ├── processed/          # Shared model-ready data
│   └── raw/                # Saved stock list and prices
├── docs/                   # Group workflow and backtest plan
├── notebooks/              # Data setup and model experiments
├── references/             # Papers and notes
├── reports/
│   └── figures/            # Charts for our results
├── src/
│   ├── backtests/          # Shared backtest code
│   ├── data/               # Data loading and cleaning
│   ├── factors/            # Features and signals
│   ├── portfolio/          # Stock selection and weights
│   └── utils/              # Shared helpers
└── tests/                  # Checks for the project code
```

## Simon's research showcase

On `SimonResearch`, open [`notebooks/SimonResearch_Showcase.ipynb`](notebooks/SimonResearch_Showcase.ipynb) or run:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-showcase.txt
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebuild-features --retrain
```

The default research run uses the same saved universe and price range as `Data.ipynb`, adds 15 literature-motivated features to its original nine, fits four fixed models, selects the blend on 2023, and evaluates 2024–2025. The [feature guide](docs/alpha-features.md) explains the 24 raw inputs and their 24 same-day percentile transforms. One-year features require a longer warm-up, so the expanded training period starts in 2017 and still ends in 2022.

The new dashboard is [`reports/alpha/index.html`](reports/alpha/index.html), with [measured results](reports/alpha/RESULTS.md), the full comparison, data hashes and fitted models saved locally. The earlier nine-feature experiment remains in [`reports/showcase/`](reports/showcase/RESULTS.md). The saved original model is also scored on the expanded experiment's eligible stocks for a matched test-universe comparison; its training history differs, so this is not a controlled feature-only ablation.

```bash
# Explicitly rebuild features from the saved prices and train again
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebuild-features --retrain
# Verify and reuse the completed research experiment
.venv/bin/python scripts/run_showcase.py --feature-set alpha
```

Later runs verify checksums and reuse the completed experiment. `--rebacktest` recomputes execution with frozen forecasts; `--retrain` explicitly repeats training; `--rebuild-features` rebuilds the new features from saved data. `--refresh-data` downloads a new snapshot. `--feature-set notebook` selects the original feature schema and its separate artifact directory. If no saved prices exist, preparation follows the notebook's download settings. Repeating experiments after seeing test results makes that window exploratory.

The [optional TabPFN-3.5 comparison](docs/tabpfn.md) uses the same 48 inputs with a fixed 20,000-row training context, explicit cloud-inference budgets, and a batch-causality check. It produces a separate candidate.

The four model candidates are XGBoost regression, XGBoost LambdaMART ranking, CatBoost and ElasticNet. Ranking scores and calibrated next-day return estimates are exported separately. The $1 million long-only backtest adds to the top 20 at the next session's OHLC4 proxy. It holds until the prior-close rank leaves the top 100. All already-available free cash funds long additions; transaction costs remain zero as requested. Every dashboard and notebook run shows each model, the blend and the saved baselines separately. See the [holding-strategy rules](docs/rank-hold-backtest.md). The target and model selection remain frozen for the long-only strategy revision. The earlier long/short alpha experiment is preserved under `reports/alpha/archive/long-short-v1/`.

The runner handles the bundled OpenMP runtime used by the macOS wheels. For direct model imports or tests on this Mac, launch Python with:

```bash
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python -m pytest -q
```

Use the `.venv` notebook kernel or install its kernel with:

```bash
.venv/bin/python -m ipykernel install --user --name feclub-research --display-name "FE Club Research"
```

Raw data, caches, the environment, fitted models, and large prediction/trade tables are ignored by Git. Small measured reports are versioned. See the [research roadmap](docs/research-roadmap.md) for the path from this demo to the eventual paper. This snapshot has current-universe survivorship bias and does not establish live trading profitability.
