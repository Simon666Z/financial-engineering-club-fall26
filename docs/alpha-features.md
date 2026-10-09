# Expanded alpha features

The alpha experiment uses the notebook's saved stock universe and price snapshot. It keeps the original nine signals and adds fifteen price/volume signals across six literature-motivated families. The priority is a reproducible model; the paper follows after we understand the evidence.

## 24 raw features become 48 model inputs

The original nine are `ret_1d`, `mom_5d`, `mom_20d`, `mom_60d`, `vol_5d`, `vol_20d`, `volume_ratio_20d`, `range_pct` and `intraday_ret`. Adding the fifteen below gives **24 raw features**.

Each raw feature also receives a same-day cross-stock rank transform, giving **24 raw + 24 transformed = 48 inputs**. For a day's eligible stocks, the transform is `2 × ((average-tie rank − 0.5) / number of stocks − 0.5)`. It centers the ordering near zero and keeps it within −1 to 1. A constant feature becomes zero. The transform uses the observable feature values for that day, including eligible rows without a future label.

The feature `dollar_volume_rank_50` is a different rank: it compares one stock's activity with its trailing 50 sessions, including today. Its additional `__csrank` model input compares that trailing-activity signal across stocks on the same date.

## The fifteen added features

Notation: `P` is adjusted close, `C` raw close, `O` raw open, `V` raw volume, `DV = C × V`, and `r[t] = P[t] / P[t-1] − 1`. Date ranges below include both endpoints. A “session” means a trading-calendar observation, not a calendar day.

The formulas and source assignments come from [ALPHA_FEATURE_METADATA](../src/data/alpha_features.py). They are explicit daily-data adaptations. The volume/reversal interaction is an implementation choice inspired by the cited literature; the 252-close high, 21-return MAX window and Open/Close decomposition also differ from the papers' exact constructions.

| Feature | Family | Implemented formula | Complete sessions | Sources |
|---|---|---|---:|---|
| `mom_126_21` | Skipped-month momentum | `P[t-21] / P[t-126] - 1` | 127 | [Momentum][mom], [French][french] |
| `mom_252_21` | Skipped-month momentum | `P[t-21] / P[t-252] - 1` | 253 | [Momentum][mom], [French][french] |
| `high_ratio_252` | 52-week high | `P[t] / max(P[t-251:t])` | 252 | [High][high] |
| `dollar_volume_rank_50` | Abnormal dollar volume | `average-tie percentile rank of DV[t] among DV[t-49:t]` | 50 | [Activity][activity] |
| `dollar_volume_log_shock_20` | Abnormal dollar volume | `log(DV[t] / mean(DV[t-20:t-1]))` | 21 | [Volume][volume] |
| `volume_reversal_1d` | Volume-conditioned reversal | `-ret_1d[t] * max(dollar_volume_log_shock_20[t], 0)` | 21 | [Volume][volume] |
| `max_return_21d` | Extreme returns | `max(ret_1d[t-20:t])` | 22 | [MAX][max] |
| `max5_return_21d` | Extreme returns | `mean(top 5 ret_1d values in t-20:t)` | 22 | [MAX][max] |
| `overnight_ret_1d` | Overnight / intraday | `(P[t]/P[t-1]) / (Close[t]/Open[t]) - 1` | 2 | [Overnight][overnight] |
| `overnight_mom_20d` | Overnight / intraday | `exp(sum(log(1 + overnight_ret_1d), last 20 sessions)) - 1` | 21 | [Overnight][overnight] |
| `intraday_mom_20d` | Overnight / intraday | `exp(sum(log(Close/Open), last 20 sessions)) - 1` | 20 | [Overnight][overnight] |
| `overnight_mom_60d` | Overnight / intraday | `exp(sum(log(1 + overnight_ret_1d), last 60 sessions)) - 1` | 61 | [Overnight][overnight] |
| `intraday_mom_60d` | Overnight / intraday | `exp(sum(log(Close/Open), last 60 sessions)) - 1` | 60 | [Overnight][overnight] |
| `overnight_intraday_spread_20d` | Overnight / intraday | `overnight_mom_20d - intraday_mom_20d` | 21 | [Overnight][overnight] |
| `overnight_intraday_spread_60d` | Overnight / intraday | `overnight_mom_60d - intraday_mom_60d` | 61 | [Overnight][overnight] |

## Availability and warmup

Every feature becomes available **after the completed close on signal date t**, using quotes through t. Future prices and training labels are not read when building inputs.

The longest feature needs **252 prior sessions plus today's observation: 253 consecutive adjusted closes**. This is the 252-session warmup recorded in the data manifest. The implementation requires the whole momentum window to be present, including the skipped recent 21 sessions. Other rolling windows also require every relevant observation. Missing stock sessions stay missing; prices are not filled and gaps are not treated as shorter trading-day lags. Eligibility requires the original nine and all fifteen additions to be finite.

These longer histories can delay the first training date and temporarily exclude stocks with incomplete histories. The recorded split dates and eligible-row counts describe the actual experiment.

## The target and trading rules continue

The learning label remains the next-day adjusted-close return, `P[t+1] / P[t] − 1`. Elastic Net, XGBoost regression and CatBoost learn its centered daily ranks; XGBoost Ranker learns daily return-decile grades. Training ends in 2022, 2023 validation selects settings and forecast weights, and the selected forecast is frozen for the 2024–2025 backtest.

The current strategy starts with $1 million, adds to the top 20 long only, and retains positions while in the top 100. Close-based decisions fill the next session using the synthetic adjusted OHLC4 proxy. All already-available free cash is allocated equally among the selected long stocks. Exits release cash for the following session's entries. The current run excludes commissions and borrowing fees. See the [holding-strategy protocol](rank-hold-backtest.md) for the complete accounting rules.

## Run and inspect

Rebuild all feature families and train a new alpha experiment:

```bash
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebuild-features --retrain
```

Reuse it with `.venv/bin/python scripts/run_showcase.py --feature-set alpha`. Prepared alpha tables and the formula/source manifest are saved under `data/processed`; models go to `models/alpha` and results to `reports/alpha`. The original `showcase` artifacts remain separate.

## Literature sources

These are the exact URLs recorded in the implementation metadata. They motivate the signal families; performance and implementation choices are evaluated in our own experiment.

- **Momentum:** [Jegadeesh and Titman (1993), Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency][mom], plus [Kenneth French's daily momentum construction][french].
- **52-week high:** [George and Hwang (2004), The 52-Week High and Momentum Investing][high].
- **Abnormal activity:** [Gervais, Kaniel and Mingelgrin, The High Volume Return Premium, working-paper version][activity].
- **Volume and reversal:** [Campbell, Grossman and Wang (1992), Trading Volume and Serial Correlation in Stock Returns, NBER Working Paper 4193][volume].
- **Extreme returns:** [Bali, Cakici and Whitelaw (2011), Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns][max].
- **Overnight/intraday:** [Lou, Polk and Skouras (2019), A Tug of War: Overnight versus Intraday Expected Returns][overnight].

[mom]: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x
[french]: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html
[high]: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x
[volume]: https://www.nber.org/papers/w4193
[activity]: https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/9901.pdf
[max]: https://pages.stern.nyu.edu/~rwhitela/papers/max%20jfe11.pdf
[overnight]: https://personal.lse.ac.uk/polk/research/TugOfWar.pdf
