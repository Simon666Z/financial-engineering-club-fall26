"""Verify preserved execution prices using local synthetic quotes only."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtests.showcase import evaluate_predictions
from src.data.preparation import prepare_price_panel
from src.data.showcase import execution_panel


def prices(sessions=80):
    dates = pd.bdate_range("2024-01-02", periods=sessions)
    rows = []
    for date in dates:
        for ticker in ["AAA", "BBB"]:
            rows.append({
                "Date": date, "Ticker": ticker, "Open": 100.0, "High": 101.0,
                "Low": 99.0, "Close": 100.0, "Adj Close": 100.0, "Volume": 1_000_000,
            })
    return pd.DataFrame(rows), dates


class ShowcaseDataTests(unittest.TestCase):
    def test_valid_open_survives_missing_close_and_becomes_unresolved_trade(self):
        raw, dates = prices()
        missing_close = raw["Ticker"].eq("AAA") & raw["Date"].eq(dates[71])
        raw.loc[missing_close, "Close"] = np.nan
        cleaned, _ = prepare_price_panel(raw)
        result = execution_panel(cleaned, raw)
        known = result.loc[result["Ticker"].eq("AAA") & result["Date"].eq(dates[71])].iloc[0]
        self.assertTrue(pd.isna(known["Open"]))
        self.assertFalse(known["QuoteValid"])
        self.assertFalse(known["SignalEligible"])
        self.assertEqual(known["ExecutionOpen"], 100.0)
        self.assertTrue(pd.isna(known["ExecutionClose"]))
        assert_frame_equal(result[cleaned.columns], cleaned)
        # The previous signal cannot know that the following close is absent.
        prediction_rows = result.loc[result["SignalEligible"] & result["Date"].ge(dates[70]), ["Date", "Ticker"]].copy()
        prediction_rows["prediction"] = prediction_rows["Ticker"].eq("AAA").astype(float)
        summary, daily, trades = evaluate_predictions(result, prediction_rows, top_k=1, start_date=dates[70])
        self.assertTrue(daily.iloc[0]["unresolved"])
        self.assertEqual(daily.iloc[0]["executed_count"], 1)
        self.assertEqual(daily.iloc[0]["missing_entry_count"], 0)
        self.assertEqual(summary["unresolved_trading_dates"], 1)
        self.assertEqual(trades.loc[trades["portfolio"].eq("strategy")].iloc[0]["status"], "unresolved_exit")

    def test_execution_prices_are_independently_numeric_finite_and_positive(self):
        rows = []
        bad = [0.0, -1.0, np.inf, -np.inf, np.nan, "not-a-number"]
        for index, value in enumerate(bad):
            rows.extend([
                {"Date": "2024-01-02", "Ticker": f"OPEN{index}", "Open": value, "Close": "123.5"},
                {"Date": "2024-01-02", "Ticker": f"CLOSE{index}", "Open": "42.25", "Close": value},
            ])
        raw = pd.DataFrame(rows)
        cleaned = raw[["Date", "Ticker"]].copy()
        cleaned["Date"] = pd.to_datetime(cleaned["Date"])
        cleaned["SignalEligible"] = False
        result = execution_panel(cleaned, raw)
        open_bad = result.loc[result["Ticker"].str.startswith("OPEN")]
        close_bad = result.loc[result["Ticker"].str.startswith("CLOSE")]
        self.assertTrue(open_bad["ExecutionOpen"].isna().all())
        self.assertTrue(open_bad["ExecutionClose"].eq(123.5).all())
        self.assertTrue(close_bad["ExecutionOpen"].eq(42.25).all())
        self.assertTrue(close_bad["ExecutionClose"].isna().all())
        self.assertFalse(result["SignalEligible"].any())

    def test_conflicting_duplicates_are_rejected_identical_quotes_collapse(self):
        raw, _ = prices(sessions=2)
        cleaned, _ = prepare_price_panel(raw)
        repeated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        result = execution_panel(cleaned, repeated)
        self.assertEqual(len(result), len(cleaned))
        conflict = raw.iloc[[0]].copy()
        conflict["Open"] = 100.5
        with self.assertRaisesRegex(ValueError, "Conflicting execution quotes"):
            execution_panel(cleaned, pd.concat([raw, conflict], ignore_index=True))
        conflict = raw.iloc[[0]].copy()
        conflict["Close"] = 100.5
        with self.assertRaisesRegex(ValueError, "Conflicting execution quotes"):
            execution_panel(cleaned, pd.concat([raw, conflict], ignore_index=True))

    def test_normalized_keys_and_absent_quotes_preserve_cleaned_calendar(self):
        raw = pd.DataFrame([
            {"Date": "2024-01-02T13:00:00Z", "Ticker": " brk.b ", "Open": 100.0, "Close": 101.0},
        ])
        cleaned = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Ticker": ["BRK-B", "BRK-B"], "SignalEligible": [True, False],
        })
        result = execution_panel(cleaned, raw)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["ExecutionOpen"], 100.0)
        self.assertEqual(result.iloc[0]["ExecutionClose"], 101.0)
        self.assertTrue(pd.isna(result.iloc[1]["ExecutionOpen"]))
        self.assertTrue(pd.isna(result.iloc[1]["ExecutionClose"]))
        assert_frame_equal(result[cleaned.columns], cleaned)


if __name__ == "__main__":
    unittest.main()
