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
.venv/bin/python scripts/run_showcase.py
```

The first run downloads the same universe and date range as `Data.ipynb`, fits four fixed models, selects the blend on 2023, and evaluates 2024–2025. It saves a dashboard at [`reports/showcase/index.html`](reports/showcase/index.html), a complete comparison, data hashes, and fitted models locally. Later runs verify checksums and reuse the completed experiment. `--rebacktest` recomputes execution with the frozen model; `--retrain` explicitly repeats training; `--refresh-data` downloads a new snapshot. Repeating experiments after seeing test results makes that window exploratory.

The model combines XGBoost regression, XGBoost LambdaMART ranking, CatBoost, and ElasticNet through a validation-selected blend. It uses the original nine features plus their observable same-day percentiles. Ranking scores and calibrated return estimates are exported separately. The $1 million backtest adds longs from the top 20 and shorts from the bottom 20 at the next session's OHLC4 proxy. It holds until the prior-close rank leaves the top/bottom 100. New available capital is split 50/50, with short proceeds and collateral reserved; transaction and borrow fees are temporarily set to zero. See the [holding-strategy rules](docs/rank-hold-backtest.md) and [measured results](reports/showcase/RESULTS.md). The first intraday experiment remains archived.

The runner handles the bundled OpenMP runtime used by the macOS wheels. For direct model imports or tests on this Mac, launch Python with:

```bash
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python -m pytest -q
```

Use the `.venv` notebook kernel or install its kernel with:

```bash
.venv/bin/python -m ipykernel install --user --name feclub-research --display-name "FE Club Research"
```

Raw data, caches, the environment, fitted models, and large prediction/trade tables are ignored by Git. Small measured reports are versioned. See the [research roadmap](docs/research-roadmap.md) for the path from this demo to the eventual paper. This snapshot has current-universe survivorship bias and does not establish live trading profitability.
