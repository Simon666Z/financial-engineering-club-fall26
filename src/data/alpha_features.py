"""Fifteen literature-motivated daily-data features using completed quotes only.

These are explicit adaptations, not exact replications of the cited papers.
No target, future label, split or later quote is read. Features are computed on
one shared observed calendar; missing stock sessions are never bridged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtests.showcase import _normalize_keys


ALPHA_FEATURES = [
    "mom_126_21", "mom_252_21", "high_ratio_252", "dollar_volume_rank_50",
    "dollar_volume_log_shock_20", "volume_reversal_1d", "max_return_21d",
    "max5_return_21d", "overnight_ret_1d", "overnight_mom_20d",
    "intraday_mom_20d", "overnight_mom_60d", "intraday_mom_60d",
    "overnight_intraday_spread_20d", "overnight_intraday_spread_60d",
]

_MOMENTUM_SOURCES = [
    "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x",
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html",
]
_HIGH_SOURCE = "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x"
_VOLUME_SOURCE = "https://www.nber.org/papers/w4193"
_DOLLAR_VOLUME_SOURCE = "https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/9901.pdf"
_MAX_SOURCE = "https://pages.stern.nyu.edu/~rwhitela/papers/max%20jfe11.pdf"
_OVERNIGHT_SOURCE = "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf"


def _spec(formula, required_sessions, sources, notes):
    return {
        "formula": formula, "required_sessions": required_sessions,
        "availability": "after completed close on signal date t",
        "source_urls": sources, "adaptation": True, "notes": notes,
    }


ALPHA_FEATURE_METADATA = {
    "mom_126_21": _spec("P[t-21] / P[t-126] - 1", 127, _MOMENTUM_SOURCES,
                        "Daily skip-month momentum; all adjusted prices from t-126 through t must be present, including the skipped recent interval."),
    "mom_252_21": _spec("P[t-21] / P[t-252] - 1", 253, _MOMENTUM_SOURCES,
                        "Daily skip-month momentum; all adjusted prices from t-252 through t must be present, including the skipped recent interval."),
    "high_ratio_252": _spec("P[t] / max(P[t-251:t])", 252, [_HIGH_SOURCE],
                           "Adjusted-close ratio to a complete 252-session high; not the paper's exact monthly construction."),
    "dollar_volume_rank_50": _spec("average-tie percentile rank of DV[t] among DV[t-49:t]", 50, [_DOLLAR_VOLUME_SOURCE],
                                 "Within-stock trailing dollar-volume percentile, including today; daily-data adaptation."),
    "dollar_volume_log_shock_20": _spec("log(DV[t] / mean(DV[t-20:t-1]))", 21, [_VOLUME_SOURCE],
                                      "Positive dollar volume; the baseline excludes today's observation."),
    "volume_reversal_1d": _spec("-ret_1d[t] * max(dollar_volume_log_shock_20[t], 0)", 21, [_VOLUME_SOURCE],
                              "Deterministic volume/reversal interaction motivated by the literature; not a formula claimed to appear in the paper."),
    "max_return_21d": _spec("max(ret_1d[t-20:t])", 22, [_MAX_SOURCE],
                           "Largest daily adjusted-close return in a complete 21-return window; daily approximation to monthly MAX."),
    "max5_return_21d": _spec("mean(top 5 ret_1d values in t-20:t)", 22, [_MAX_SOURCE],
                            "Mean of the five largest returns in the same complete window; daily approximation to MAX5."),
    "overnight_ret_1d": _spec("(P[t]/P[t-1]) / (Close[t]/Open[t]) - 1", 2, [_OVERNIGHT_SOURCE],
                             "Adjusted total return decomposed using raw open/close, not opening VWAP; corporate-action effects follow the supplied adjustment basis."),
}
for _window in [20, 60]:
    ALPHA_FEATURE_METADATA[f"overnight_mom_{_window}d"] = _spec(
        f"exp(sum(log(1 + overnight_ret_1d), last {_window} sessions)) - 1", _window + 1, [_OVERNIGHT_SOURCE],
        "Complete compounded overnight-component window; open-based daily-data adaptation.",
    )
    ALPHA_FEATURE_METADATA[f"intraday_mom_{_window}d"] = _spec(
        f"exp(sum(log(Close/Open), last {_window} sessions)) - 1", _window, [_OVERNIGHT_SOURCE],
        "Complete compounded intraday-component window; open-based daily-data adaptation.",
    )
    ALPHA_FEATURE_METADATA[f"overnight_intraday_spread_{_window}d"] = _spec(
        f"overnight_mom_{_window}d - intraday_mom_{_window}d", _window + 1, [_OVERNIGHT_SOURCE],
        "Difference of separately compounded components, not a difference of daily arithmetic averages.",
    )


def _rolling(values, window, operation):
    return values.groupby(level="Ticker", sort=False).transform(
        lambda series: getattr(series.rolling(window, min_periods=window), operation)()
    )


def _top_five_mean(values):
    return float(np.partition(values, -5)[-5:].mean())


def _positive(values):
    values = pd.to_numeric(values, errors="coerce").astype(float)
    return values.where(np.isfinite(values) & values.gt(0))


def add_alpha_features(panel):
    """Return a copy with 15 causal features and AlphaSignalEligible.

    Existing rows, their order/index and all original columns are preserved.
    Keys are normalized only for internal grouping and validation. Conflicting
    or repeated normalized date/ticker keys and malformed keys are rejected.
    Each stock is internally aligned to every supplied shared calendar date;
    an omitted stock row remains a missing session rather than a shorter lag.

    Momentum requires the entire horizon+1 price window, including its recent
    skipped interval. Rolling features require every relevant observation.
    No missing value is filled. Eligibility combines the original known-t
    SignalEligible flag with finite values for every new feature.

    P is positive finite adjusted close and DV is positive finite raw close
    times raw volume. Returns use the supplied ret_1d only where both adjacent
    adjusted closes and a finite return greater than -1 are available.
    """
    required = {"Date", "Ticker", "Adj Close", "Open", "Close", "Volume", "ret_1d", "SignalEligible"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Alpha feature panel is missing columns: {sorted(missing)}")
    clean = _normalize_keys(panel[list(required)], "Alpha feature panel")
    if clean["SignalEligible"].isna().any() or not clean["SignalEligible"].isin([True, False]).all():
        raise ValueError("SignalEligible must contain nonmissing boolean values.")
    result = panel.copy()
    if clean.empty:
        for name in ALPHA_FEATURES:
            result[name] = pd.Series(index=result.index, dtype=float)
        result["AlphaSignalEligible"] = pd.Series(index=result.index, dtype=bool)
        return result
    dates = pd.DatetimeIndex(clean["Date"].unique()).sort_values()
    tickers = sorted(clean["Ticker"].unique())
    grid = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])
    values = clean.set_index(["Ticker", "Date"]).reindex(grid)
    price, open_price, close_price, volume = [_positive(values[name]) for name in ["Adj Close", "Open", "Close", "Volume"]]
    previous = price.groupby(level="Ticker", sort=False).shift(1)
    returns = pd.to_numeric(values["ret_1d"], errors="coerce").astype(float)
    returns = returns.where(np.isfinite(returns) & returns.gt(-1) & price.notna() & previous.notna())
    features = pd.DataFrame(index=grid)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        for horizon in [126, 252]:
            endpoint = price.groupby(level="Ticker", sort=False).shift(horizon)
            recent = price.groupby(level="Ticker", sort=False).shift(21)
            complete = _rolling(price, horizon + 1, "count").eq(horizon + 1)
            features[f"mom_{horizon}_21"] = (recent / endpoint - 1).where(complete)
        features["high_ratio_252"] = price / _rolling(price, 252, "max")
        dollar_volume = close_price * volume
        dollar_volume = dollar_volume.where(np.isfinite(dollar_volume) & dollar_volume.gt(0))
        features["dollar_volume_rank_50"] = dollar_volume.groupby(level="Ticker", sort=False).transform(
            lambda series: series.rolling(50, min_periods=50).rank(method="average", pct=True)
        )
        past_volume = dollar_volume.groupby(level="Ticker", sort=False).shift(1)
        features["dollar_volume_log_shock_20"] = np.log(dollar_volume / _rolling(past_volume, 20, "mean"))
        features["volume_reversal_1d"] = -returns * features["dollar_volume_log_shock_20"].clip(lower=0)
        features["max_return_21d"] = _rolling(returns, 21, "max")
        features["max5_return_21d"] = returns.groupby(level="Ticker", sort=False).transform(
            lambda series: series.rolling(21, min_periods=21).apply(_top_five_mean, raw=True)
        )
        intraday_factor = close_price / open_price
        intraday_factor = intraday_factor.where(np.isfinite(intraday_factor) & intraday_factor.gt(0))
        intraday_log = np.log(intraday_factor)
        total_factor = price / previous
        total_factor = total_factor.where(np.isfinite(total_factor) & total_factor.gt(0))
        overnight_log = np.log(total_factor) - intraday_log
        features["overnight_ret_1d"] = np.expm1(overnight_log)
        for window in [20, 60]:
            overnight = np.expm1(_rolling(overnight_log, window, "sum"))
            intraday = np.expm1(_rolling(intraday_log, window, "sum"))
            features[f"overnight_mom_{window}d"] = overnight
            features[f"intraday_mom_{window}d"] = intraday
            features[f"overnight_intraday_spread_{window}d"] = overnight - intraday
    features = features[ALPHA_FEATURES].replace([np.inf, -np.inf], np.nan)
    original_keys = pd.MultiIndex.from_frame(clean[["Ticker", "Date"]])
    aligned = features.reindex(original_keys)
    result[ALPHA_FEATURES] = aligned.to_numpy(dtype=float)
    result["AlphaSignalEligible"] = clean["SignalEligible"].to_numpy(dtype=bool) & np.isfinite(aligned.to_numpy(dtype=float)).all(axis=1)
    return result
