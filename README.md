# financial-engineering-club-fall26

## Starter notebook

Start with [`notebooks/Data.ipynb`](notebooks/Data.ipynb) to download S&P 500 daily prices, build simple price and volume features, and create a next-day return target for machine learning exercises.

Install Jupyter and the imported packages in the same Python environment, then open the notebook:

```bash
python -m pip install --upgrade jupyter pandas numpy requests matplotlib yfinance pyarrow lxml xgboost scikit-learn
jupyter notebook notebooks/Data.ipynb
```

Run the cells from top to bottom. Internet access is required for the Wikipedia constituent list and Yahoo Finance downloads. The notebook filters out unavailable tickers. Use a current pandas release for its `future_stack=True` option.

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
