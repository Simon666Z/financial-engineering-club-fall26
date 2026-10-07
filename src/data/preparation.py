"""Clean daily quotes and create the club's shared, causal learning dataset.

All functions operate on DataFrames. They never download data or write files.
The shared session grid deliberately retains missing quotes: a shift of one
session must mean the next market date, not the next available stock quote.
"""

import numpy as np
import pandas as pd


FEATURES = [
    "ret_1d", "mom_5d", "mom_20d", "mom_60d", "vol_5d", "vol_20d",
    "volume_ratio_20d", "range_pct", "intraday_ret",
]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
QUOTE_COLUMNS = PRICE_COLUMNS + ["Volume"]


def _dates(values):
    """Use normalized, timezone-free dates as the daily join keys."""
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def _tickers(values):
    return values.astype("string").str.strip().str.upper().str.replace(".", "-", regex=False)


def prepare_price_panel(raw_panel, metadata=None):
    """Return a complete Date x Ticker panel and a row-count cleaning audit.

    Conflicting quotes for the same normalized key are errors. Invalid quotes
    remain in the calendar but their prices and volume become NaN. Reason
    counts overlap, so they must not be summed to count invalid rows.
    """
    required = ["Date", "Ticker"] + QUOTE_COLUMNS
    missing = sorted(set(required) - set(raw_panel.columns))
    if missing:
        raise ValueError(f"Missing required quote columns: {missing}")

    quotes = raw_panel[required].copy()
    quotes["Date"] = _dates(quotes["Date"])
    quotes["Ticker"] = _tickers(quotes["Ticker"])
    bad_key = quotes["Date"].isna() | quotes["Ticker"].isna() | quotes["Ticker"].eq("")
    audit = {"input_rows": len(quotes), "dropped_invalid_keys": int(bad_key.sum())}
    quotes = quotes.loc[~bad_key].copy()
    if quotes.empty:
        raise ValueError("No rows have a valid Date and Ticker")
    raw_dates = sorted(quotes["Date"].unique())
    raw_tickers = sorted(quotes["Ticker"].unique())

    # Normalize numerical strings and non-finite values before comparison.
    quotes[QUOTE_COLUMNS] = quotes[QUOTE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    quotes[QUOTE_COLUMNS] = quotes[QUOTE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    before = len(quotes)
    quotes = quotes.drop_duplicates(subset=required)
    audit["duplicate_rows_removed"] = before - len(quotes)
    conflicts = quotes.duplicated(["Date", "Ticker"], keep=False)
    if conflicts.any():
        example = quotes.loc[conflicts, ["Date", "Ticker"]].iloc[0].to_dict()
        raise ValueError(f"Conflicting quotes for a Date/Ticker key: {example}")

    # Build the grid before quote masking, including tickers with no good rows.
    grid = pd.MultiIndex.from_product(
        [raw_tickers, raw_dates],
        names=["Ticker", "Date"],
    )
    panel = quotes.set_index(["Ticker", "Date"]).reindex(grid).reset_index()
    audit["grid_rows"] = len(panel)
    audit["inserted_missing_rows"] = len(panel) - len(quotes)

    incomplete = panel[QUOTE_COLUMNS].isna().any(axis=1)
    nonpositive_price = panel[PRICE_COLUMNS].le(0).any(axis=1)
    nonpositive_volume = panel["Volume"].le(0)
    bad_ohlc = (
        panel["High"].lt(panel[["Open", "Low", "Close"]].max(axis=1))
        | panel["Low"].gt(panel[["Open", "High", "Close"]].min(axis=1))
    )
    panel["QuoteValid"] = ~(incomplete | nonpositive_price | nonpositive_volume | bad_ohlc)
    panel.loc[~panel["QuoteValid"], QUOTE_COLUMNS] = np.nan

    if metadata is not None:
        if "Ticker" not in metadata:
            raise ValueError("Metadata must have a Ticker column")
        meta = metadata.copy()
        meta["Ticker"] = _tickers(meta["Ticker"])
        meta = meta.drop_duplicates()
        if meta["Ticker"].isna().any() or meta["Ticker"].eq("").any():
            raise ValueError("Metadata contains a missing Ticker")
        if meta["Ticker"].duplicated().any():
            raise ValueError("Metadata must have one record per normalized Ticker")
        collision = sorted((set(meta.columns) & set(panel.columns)) - {"Ticker"})
        if collision:
            raise ValueError(f"Metadata overlaps quote columns: {collision}")
        panel = panel.merge(meta, on="Ticker", how="left", validate="many_to_one", sort=False)

    # All rolling calculations require complete windows; nothing is forward-filled.
    groups = panel.groupby("Ticker", sort=False)
    for horizon, name in [(1, "ret_1d"), (5, "mom_5d"), (20, "mom_20d"), (60, "mom_60d")]:
        panel[name] = groups["Adj Close"].pct_change(horizon, fill_method=None)
    for window in [5, 20]:
        panel[f"vol_{window}d"] = panel.groupby("Ticker", sort=False)["ret_1d"].transform(
            lambda values: values.rolling(window, min_periods=window).std()
        )
    panel["volume_ma20"] = groups["Volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    panel["volume_ratio_20d"] = panel["Volume"] / panel["volume_ma20"]
    panel["range_pct"] = (panel["High"] - panel["Low"]) / panel["Close"]
    panel["intraday_ret"] = panel["Close"] / panel["Open"] - 1
    panel[FEATURES] = panel[FEATURES].replace([np.inf, -np.inf], np.nan)
    panel["SignalEligible"] = panel["QuoteValid"] & panel[FEATURES].notna().all(axis=1)

    groups = panel.groupby("Ticker", sort=False)
    # This is a learning label, not a promise of executable close-to-close trades.
    panel["LabelEndDate"] = groups["Date"].shift(-1)
    panel["target"] = groups["Adj Close"].shift(-1) / panel["Adj Close"] - 1
    panel["target"] = panel["target"].replace([np.inf, -np.inf], np.nan)

    audit["valid_quote_rows"] = int(panel["QuoteValid"].sum())
    audit["invalid_quote_rows"] = int((~panel["QuoteValid"]).sum())
    audit["signal_eligible_rows"] = int(panel["SignalEligible"].sum())
    audit["invalid_reason_counts"] = {
        "missing_or_nonfinite_quote": int(incomplete.sum()),
        "nonpositive_price": int(nonpositive_price.sum()),
        "nonpositive_volume": int(nonpositive_volume.sum()),
        "invalid_ohlc_bounds": int(bad_ohlc.sum()),
    }
    audit["reason_counts_overlap"] = True
    return panel.sort_values(["Date", "Ticker"]).reset_index(drop=True), audit


def make_model_dataset(panel, *, labeled_only=True):
    """Select usable signals and rank observed returns within each market date.

    SignalEligible depends only on current/past information. This separate
    table defaults to requiring a known finite future return for learning.
    Set labeled_only=False for a shared feature export; learners must drop
    missing target values before training or evaluating their model.
    Higher percentiles mean higher realized returns; ties receive their average.
    """
    values = pd.to_numeric(panel["target"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = panel["SignalEligible"]
    if labeled_only:
        keep = keep & values.notna()
    dataset = panel.loc[keep].copy()
    dataset["target"] = values.loc[keep]
    dataset["target_rank"] = dataset.groupby("Date")["target"].rank(method="average", pct=True)
    return dataset.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def add_date_splits(dataset, train_end="2022-12-31", validation_end="2023-12-31"):
    """Assign chronological splits, purging labels that cross either cutoff.

    Purge an entire signal date if any row's label crosses that split's boundary.
    This keeps every stock on a date together and prevents future-return leakage.
    """
    train_cutoff = pd.Timestamp(train_end).normalize()
    validation_cutoff = pd.Timestamp(validation_end).normalize()
    if pd.isna(train_cutoff) or pd.isna(validation_cutoff) or validation_cutoff <= train_cutoff:
        raise ValueError("validation_end must be later than train_end")
    result = dataset.copy()
    result["Date"] = _dates(result["Date"])
    result["LabelEndDate"] = _dates(result["LabelEndDate"])
    if result["Date"].isna().any():
        raise ValueError("Dataset contains a missing Date")
    result["Split"] = np.select(
        [result["Date"].le(train_cutoff), result["Date"].le(validation_cutoff)],
        ["train", "validation"], default="test",
    )
    crossing = (
        (result["Split"].eq("train") & result["LabelEndDate"].gt(train_cutoff))
        | (result["Split"].eq("validation") & result["LabelEndDate"].gt(validation_cutoff))
        | (result["Split"].ne("test") & result["LabelEndDate"].isna())
    )
    result.loc[result["Date"].isin(result.loc[crossing, "Date"]), "Split"] = "purged"
    return result.sort_values(["Date", "Ticker"]).reset_index(drop=True)
