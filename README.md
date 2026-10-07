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

## Use the same data

The first run needs internet. Later runs reuse the stock list and prices in `data/raw/`. Keep these files for the backtest.

The leader shares one `data/processed/model_data.csv` with everyone. Compare its printed checksum — a file fingerprint — to check everyone has the same snapshot. Generated data stays out of Git.

Use only the notebook's nine listed features as inputs. Predict `target`, the next-day adjusted-close return, then rank predictions within each date. Fit scaling and other learned data steps on training rows only.

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
