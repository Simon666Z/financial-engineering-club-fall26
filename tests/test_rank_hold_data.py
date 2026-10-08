"""Keep execution/marks separate from feature eligibility and future quotes."""

import numpy as np
import pandas as pd
import pytest

from src.data.preparation import FEATURES, prepare_price_panel
from src.data.rank_hold import add_rank_hold_prices


FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def raw_prices(days=3, tickers=("ALFA", "BETA")):
    dates = pd.bdate_range("2024-01-02", periods=days)
    columns = pd.MultiIndex.from_product([tickers, FIELDS], names=["Ticker", "Field"])
    raw = pd.DataFrame(index=dates, columns=columns, dtype=float)
    for number, ticker in enumerate(tickers):
        base = 100.0 + 25 * number + np.arange(days) * 0.2
        raw[(ticker, "Open")] = base
        raw[(ticker, "High")] = base + 3.0
        raw[(ticker, "Low")] = base - 2.0
        raw[(ticker, "Close")] = base + 1.0
        raw[(ticker, "Adj Close")] = (base + 1.0) * 0.8
        raw[(ticker, "Volume")] = 1e6
    return raw


def panel_keys(raw):
    keys = pd.MultiIndex.from_product(
        [raw.index, raw.columns.get_level_values("Ticker").unique()], names=["Date", "Ticker"]
    )
    panel = keys.to_frame(index=False)
    panel["SignalEligible"] = True
    panel["target"] = np.nan
    for number, feature in enumerate(FEATURES):
        panel[feature] = (number + 1) * 0.01
    return panel


def test_exact_ohlc4_and_adjustment_factor():
    raw = raw_prices(days=1, tickers=("ALFA",))
    raw.loc[raw.index[0], ("ALFA", "Open")] = 100.0
    raw.loc[raw.index[0], ("ALFA", "High")] = 110.0
    raw.loc[raw.index[0], ("ALFA", "Low")] = 90.0
    raw.loc[raw.index[0], ("ALFA", "Close")] = 104.0
    raw.loc[raw.index[0], ("ALFA", "Adj Close")] = 83.2
    row = add_rank_hold_prices(panel_keys(raw), raw).iloc[0]
    assert row["RawExecutionAverage"] == pytest.approx(101.0)
    assert row["AdjustmentFactor"] == pytest.approx(0.8)
    assert row["ExecutionAverage"] == pytest.approx(80.8)
    assert row["MarkClose"] == pytest.approx(83.2)


def test_missing_volume_and_features_do_not_discard_valid_marks_or_fills():
    raw = raw_prices(days=1, tickers=("ALFA",))
    raw[("ALFA", "Volume")] = np.nan
    panel = panel_keys(raw)
    panel["SignalEligible"] = False
    panel["QuoteValid"] = False
    panel[FEATURES] = np.nan
    result = add_rank_hold_prices(panel, raw)
    assert result["MarkClose"].notna().all()
    assert result["ExecutionAverage"].notna().all()
    assert not result["SignalEligible"].any()
    assert result[FEATURES].isna().all().all()


@pytest.mark.parametrize("field", ["Open", "High", "Low", "Close"])
def test_all_four_ohlc_quotes_are_required_but_mark_is_independent(field):
    raw = raw_prices(days=3, tickers=("ALFA",))
    raw.loc[raw.index[1], ("ALFA", field)] = np.nan
    result = add_rank_hold_prices(panel_keys(raw), raw).set_index("Date")
    missing_day = result.loc[raw.index[1]]
    assert pd.isna(missing_day["RawExecutionAverage"])
    assert pd.isna(missing_day["ExecutionAverage"])
    assert missing_day["MarkClose"] == pytest.approx(raw.loc[raw.index[1], ("ALFA", "Adj Close")])
    assert result.loc[raw.index[0], "ExecutionAverage"] == pytest.approx(100.5 * 0.8)
    assert pd.notna(result.loc[raw.index[2], "ExecutionAverage"])


def test_missing_date_is_not_filled_from_a_later_or_earlier_quote():
    raw = raw_prices(days=3, tickers=("ALFA",))
    panel = panel_keys(raw)
    supplied = raw.drop(index=raw.index[1])
    result = add_rank_hold_prices(panel, supplied).set_index("Date")
    columns = ["RawExecutionAverage", "AdjustmentFactor", "ExecutionAverage", "MarkClose"]
    assert result.loc[raw.index[1], columns].isna().all()
    changed_future = supplied.copy()
    changed_future.loc[raw.index[2], ("ALFA", "Close")] = 999999.0
    changed_future.loc[raw.index[2], ("ALFA", "Adj Close")] = 888888.0
    other = add_rank_hold_prices(panel, changed_future).set_index("Date")
    pd.testing.assert_series_equal(result.loc[raw.index[1]], other.loc[raw.index[1]])
    pd.testing.assert_series_equal(result.loc[raw.index[0]], other.loc[raw.index[0]])


@pytest.mark.parametrize("field,bad_value", [
    ("Open", 0.0), ("Close", -1.0), ("High", 95.0), ("Low", 110.0),
    ("Open", np.inf), ("Adj Close", 0.0), ("Adj Close", -10.0),
])
def test_nonpositive_nonfinite_or_inconsistent_bar_cannot_be_a_fill(field, bad_value):
    raw = raw_prices(days=1, tickers=("ALFA",))
    raw.loc[raw.index[0], ("ALFA", field)] = bad_value
    row = add_rank_hold_prices(panel_keys(raw), raw).iloc[0]
    assert pd.isna(row["ExecutionAverage"])
    if field != "Adj Close":
        assert pd.notna(row["MarkClose"])
    else:
        assert pd.isna(row["MarkClose"])


def test_duplicate_raw_calendar_dates_are_rejected():
    raw = raw_prices(days=2)
    panel = panel_keys(raw)
    duplicate = pd.concat([raw, raw.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate calendar dates"):
        add_rank_hold_prices(panel, duplicate)


def test_ticker_normalization_collisions_are_rejected():
    raw = raw_prices(days=2, tickers=("A.B", "A-B"))
    panel = pd.DataFrame({"Date": raw.index, "Ticker": "A-B"})
    with pytest.raises(ValueError, match="conflicting ticker/date keys"):
        add_rank_hold_prices(panel, raw)


def test_duplicate_panel_positions_are_rejected():
    raw = raw_prices(days=1)
    panel = panel_keys(raw)
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        add_rank_hold_prices(duplicate, raw)


def test_price_basis_rebasing_changes_units_without_changing_signals_or_values():
    raw = raw_prices(days=65)
    for ticker in ("ALFA", "BETA"):
        steps = np.arange(len(raw))
        raw[(ticker, "Close")] *= 1 + 0.002 * np.sin(steps)
        raw[(ticker, "Adj Close")] = 0.8 * raw[(ticker, "Close")]
        raw[(ticker, "Volume")] += 1e4 * np.cos(steps)

    def prepared(prices):
        tidy = prices.stack(level="Ticker", future_stack=True).rename_axis(["Date", "Ticker"]).reset_index()
        panel, _ = prepare_price_panel(tidy)
        return add_rank_hold_prices(panel, prices)

    expected = prepared(raw)
    rebased = raw.copy()
    factor = 7.0
    for field in FIELDS[:-1]:
        rebased[("ALFA", field)] *= factor
    observed = prepared(rebased)
    assert expected["SignalEligible"].equals(observed["SignalEligible"])
    np.testing.assert_allclose(expected[FEATURES], observed[FEATURES], equal_nan=True, atol=1e-12)
    np.testing.assert_allclose(expected["target"], observed["target"], equal_nan=True, atol=1e-12)
    alfa = expected["Ticker"].eq("ALFA")
    np.testing.assert_allclose(observed.loc[alfa, "AdjustmentFactor"], expected.loc[alfa, "AdjustmentFactor"])
    np.testing.assert_allclose(observed.loc[alfa, "ExecutionAverage"], expected.loc[alfa, "ExecutionAverage"] * factor)
    # Dollar order -> synthetic units -> marked dollar value is basis invariant.
    for frame in (expected, observed):
        asset = frame.loc[alfa].reset_index(drop=True)
        units = 25000.0 / asset.loc[0, "ExecutionAverage"]
        marked_value = units * asset["MarkClose"]
        if frame is expected:
            original_marked_value = marked_value
        else:
            np.testing.assert_allclose(marked_value, original_marked_value)
    beta = expected["Ticker"].eq("BETA")
    pd.testing.assert_series_equal(observed.loc[beta, "ExecutionAverage"], expected.loc[beta, "ExecutionAverage"])


def test_adjusted_units_embed_distribution_without_added_cash_credits():
    raw = raw_prices(days=2, tickers=("ALFA",))
    for field in ["Open", "High", "Low", "Close"]:
        raw[("ALFA", field)] = [100.0, 90.0]
    raw[("ALFA", "Adj Close")] = [90.0, 90.0]
    panel = panel_keys(raw)
    expected = add_rank_hold_prices(panel, raw)
    assert expected["RawExecutionAverage"].tolist() == [100.0, 90.0]
    assert expected["AdjustmentFactor"].tolist() == [0.9, 1.0]
    assert expected["ExecutionAverage"].tolist() == [90.0, 90.0]
    assert expected["MarkClose"].tolist() == [90.0, 90.0]
    raw[("ALFA", "Dividends")] = [0.0, 10.0]
    raw[("ALFA", "Stock Splits")] = [0.0, 2.0]
    actual = add_rank_hold_prices(panel, raw)
    pd.testing.assert_frame_equal(expected, actual)
    assert not any("Dividend" in column or "Distribution" in column for column in actual)
    long_units, short_units = 1000 / 90, -1000 / 90
    assert long_units * (actual.loc[1, "MarkClose"] - actual.loc[0, "MarkClose"]) == 0.0
    assert short_units * (actual.loc[1, "MarkClose"] - actual.loc[0, "MarkClose"]) == 0.0
