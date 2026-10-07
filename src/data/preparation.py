"""Clean stock prices and build our shared ML table.

Keep missing dates so next-day returns cover just one trading day.
These helpers work on tables; the notebook downloads and saves the data.
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
    """Use plain dates without times or timezones."""
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def _tickers(values):
    return values.astype("string").str.strip().str.upper().str.replace(".", "-", regex=False)


def prepare_price_panel(raw_panel, metadata=None):
    """Clean daily prices and return a table plus check counts.

    Keep one row per date and ticker, including missing prices. Conflicting
    duplicates raise an error. Bad prices and volume become NaN. One row can
    fail several checks, so don't add the reason counts to get a total.
    """
    required = ["Date", "Ticker"] + QUOTE_COLUMNS
    missing = sorted(set(required) - set(raw_panel.columns))
    if missing:
        raise ValueError(f"Price table is missing these columns: {missing}")

    quotes = raw_panel[required].copy()
    quotes["Date"] = _dates(quotes["Date"])
    quotes["Ticker"] = _tickers(quotes["Ticker"])
    bad_key = quotes["Date"].isna() | quotes["Ticker"].isna() | quotes["Ticker"].eq("")
    audit = {"input_rows": len(quotes), "dropped_invalid_keys": int(bad_key.sum())}
    quotes = quotes.loc[~bad_key].copy()
    if quotes.empty:
        raise ValueError("No rows have both a valid Date and Ticker.")
    raw_dates = sorted(quotes["Date"].unique())
    raw_tickers = sorted(quotes["Ticker"].unique())

    # Read numbers from text and turn infinity into missing values.
    quotes[QUOTE_COLUMNS] = quotes[QUOTE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    quotes[QUOTE_COLUMNS] = quotes[QUOTE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    before = len(quotes)
    quotes = quotes.drop_duplicates(subset=required)
    audit["duplicate_rows_removed"] = before - len(quotes)
    conflicts = quotes.duplicated(["Date", "Ticker"], keep=False)
    if conflicts.any():
        example = quotes.loc[conflicts, ["Date", "Ticker"]].iloc[0].to_dict()
        raise ValueError(f"Conflicting quotes for this date and ticker: {example}")

    # Keep every trading date, even when a stock has no price for it.
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
            raise ValueError("Stock details need a Ticker column.")
        meta = metadata.copy()
        meta["Ticker"] = _tickers(meta["Ticker"])
        meta = meta.drop_duplicates()
        if meta["Ticker"].isna().any() or meta["Ticker"].eq("").any():
            raise ValueError("Stock details contain a missing ticker.")
        if meta["Ticker"].duplicated().any():
            raise ValueError("Stock details need one record per ticker.")
        collision = sorted((set(meta.columns) & set(panel.columns)) - {"Ticker"})
        if collision:
            raise ValueError(f"Stock details reuse these price column names: {collision}")
        panel = panel.merge(meta, on="Ticker", how="left", validate="many_to_one", sort=False)

    # Rolling averages and volatility need a full window. Leave price gaps empty.
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
    # Tomorrow's return is the learning target. Trades use separate entry and exit prices.
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
    """Keep usable feature rows and rank each day's known returns.

    SignalEligible uses only today's and past data. By default, keep rows
    with a known target. Set labeled_only=False to also keep prediction rows
    without tomorrow's return; drop missing targets when training or scoring.
    Higher ranks mean higher returns. Ties share their average rank.
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
    """Split by date into train, validation, and test.

    Mark a day 'purged' if a label falls outside its period. All stocks on
    that day stay together, keeping later returns out of earlier training.
    """
    train_cutoff = pd.Timestamp(train_end).normalize()
    validation_cutoff = pd.Timestamp(validation_end).normalize()
    if pd.isna(train_cutoff) or pd.isna(validation_cutoff) or validation_cutoff <= train_cutoff:
        raise ValueError("validation_end must come after train_end.")
    result = dataset.copy()
    result["Date"] = _dates(result["Date"])
    result["LabelEndDate"] = _dates(result["LabelEndDate"])
    if result["Date"].isna().any():
        raise ValueError("A row is missing its Date.")
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
