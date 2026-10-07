# financial-engineering-club-fall26

## Starter notebook

Run [`notebooks/Data.ipynb`](notebooks/Data.ipynb) to prepare the group's shared dataset. It keeps the original nine features and next-day return target, validates daily quotes, and creates common chronological splits: training through 2022, validation in 2023, and testing in 2024–2025.

From the cloned repository, install the data dependencies and open Jupyter:

```bash
python -m pip install jupyter -r requirements-data.txt
jupyter notebook notebooks/Data.ipynb
```

In Colab, first run `!git clone https://github.com/Simon666Z/financial-engineering-club-fall26.git`, then `%cd financial-engineering-club-fall26`. Run the notebook cells from top to bottom.

The first run needs internet access. Later runs reuse the stock list and raw prices in `data/raw/`. The leader should share the same generated `data/processed/model_data.csv` with every member; the notebook prints its checksum. Generated data is excluded from Git. Keep cached raw quotes for later portfolio evaluation.

Each member uses only the nine named features, fits a regression model on labeled training rows, and ranks predictions within each date. Fit preprocessing on training data only. The exported table keeps usable prediction rows even when their future target is unavailable; omit those rows only for fitting or scoring.

The constituent list comes from the download snapshot, not historical S&P 500 membership. The close-to-close target is a learning label; executable portfolio returns require a separate holding-period calculation. See the [simple shared backtest proposal](docs/backtesting.md).

Run the data checks with:

```bash
python -m unittest discover -s tests -v
```

## Factor Investing Project Structure

```text
.
├── config/                 # Config files (parameters, paths, settings)
├── data/
│   ├── external/           # Third-party/reference datasets
│   ├── interim/            # Intermediate transformed datasets
│   ├── processed/          # Final model-ready datasets
│   └── raw/                # Original immutable source data
├── notebooks/              # Research and exploratory analysis notebooks
├── references/             # Papers, notes, and supporting materials
├── reports/
│   └── figures/            # Generated charts and report graphics
├── src/
│   ├── backtests/          # Backtesting logic
│   ├── data/               # Data loading and preprocessing
│   ├── factors/            # Factor construction and signals
│   ├── portfolio/          # Portfolio construction and optimization
│   └── utils/              # Shared utilities
└── tests/                  # Tests for project code
```
