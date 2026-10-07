"""Offline regression checks for the shared data preparation contract."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preparation import (
    FEATURES, add_date_splits, make_model_dataset, prepare_price_panel,
)


def synthetic_quotes(sessions=100):
    dates = pd.bdate_range("2022-09-01", periods=sessions)
    rows = []
    for ticker, slope in [("AAA", 0.002), ("BBB", -0.001)]:
        for i, date in enumerate(dates):
            price = 100 * np.exp(slope * i + 0.001 * np.sin(i))
            rows.append({
                "Date": date, "Ticker": ticker, "Open": price * 0.999,
                "High": price * 1.01, "Low": price * 0.99, "Close": price,
                "Adj Close": price * 0.8, "Volume": 1_000_000 + 1_000 * i,
            })
    return pd.DataFrame(rows), dates


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.dates = synthetic_quotes()

    def row(self, panel, ticker, session):
        return panel.loc[panel["Ticker"].eq(ticker) & panel["Date"].eq(self.dates[session])].iloc[0]

    def test_valid_features_labels_and_grid(self):
        panel, audit = prepare_price_panel(self.raw)
        row = self.row(panel, "AAA", 70)
        prices = self.raw.loc[self.raw["Ticker"].eq("AAA")].reset_index(drop=True)
        self.assertEqual(len(panel), 200)
        self.assertEqual(audit["valid_quote_rows"], 200)
        self.assertTrue(row["SignalEligible"])
        self.assertAlmostEqual(row["mom_60d"], prices.loc[70, "Adj Close"] / prices.loc[10, "Adj Close"] - 1)
        returns = prices["Adj Close"].pct_change(fill_method=None)
        self.assertAlmostEqual(row["vol_20d"], returns.iloc[51:71].std())
        self.assertAlmostEqual(row["volume_ratio_20d"], prices.loc[70, "Volume"] / prices.loc[51:70, "Volume"].mean())
        self.assertAlmostEqual(row["target"], prices.loc[71, "Adj Close"] / prices.loc[70, "Adj Close"] - 1)
        self.assertEqual(row["LabelEndDate"], self.dates[71])
        self.assertFalse(self.row(panel, "AAA", 59)["SignalEligible"])
        self.assertTrue(self.row(panel, "AAA", 60)["SignalEligible"])

    def test_invalid_quotes_are_masked_and_counted(self):
        for ticker, session, column, value in [
            ("AAA", 70, "High", 1.0), ("BBB", 71, "Volume", 0),
            ("AAA", 72, "Close", np.inf), ("BBB", 73, "Adj Close", -1.0),
        ]:
            mask = self.raw["Ticker"].eq(ticker) & self.raw["Date"].eq(self.dates[session])
            self.raw.loc[mask, column] = value
        panel, audit = prepare_price_panel(self.raw)
        self.assertEqual(audit["invalid_quote_rows"], 4)
        self.assertTrue(audit["reason_counts_overlap"])
        for ticker, session in [("AAA", 70), ("BBB", 71), ("AAA", 72), ("BBB", 73)]:
            row = self.row(panel, ticker, session)
            self.assertFalse(row["QuoteValid"])
            self.assertFalse(row["SignalEligible"])
            self.assertTrue(row[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].isna().all())
        self.assertGreaterEqual(audit["invalid_reason_counts"]["invalid_ohlc_bounds"], 1)
        self.assertEqual(audit["invalid_reason_counts"]["nonpositive_volume"], 1)
        self.assertEqual(audit["invalid_reason_counts"]["missing_or_nonfinite_quote"], 1)

    def test_missing_quote_does_not_bridge_sessions(self):
        missing = self.raw["Ticker"].eq("AAA") & self.raw["Date"].eq(self.dates[71])
        panel, audit = prepare_price_panel(self.raw.loc[~missing])
        self.assertEqual(audit["inserted_missing_rows"], 1)
        self.assertEqual(len(panel), 200)
        self.assertFalse(self.row(panel, "AAA", 71)["QuoteValid"])
        self.assertTrue(pd.isna(self.row(panel, "AAA", 72)["ret_1d"]))
        self.assertTrue(pd.isna(self.row(panel, "AAA", 70)["target"]))
        self.assertTrue(pd.isna(self.row(panel, "AAA", 71)["target"]))
        self.assertEqual(self.row(panel, "AAA", 70)["LabelEndDate"], self.dates[71])
        self.assertTrue(pd.isna(self.row(panel, "AAA", 73)["vol_5d"]))

    def test_identical_duplicates_removed_conflicts_rejected(self):
        panel, audit = prepare_price_panel(pd.concat([self.raw, self.raw.iloc[[70]]], ignore_index=True))
        self.assertEqual(audit["duplicate_rows_removed"], 1)
        self.assertEqual(len(panel), len(self.raw))
        conflict = self.raw.iloc[[70]].copy()
        conflict["Volume"] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting quotes"):
            prepare_price_panel(pd.concat([self.raw, conflict], ignore_index=True))

    def test_invalid_keys_cannot_add_phantom_sessions_or_tickers(self):
        expected, _ = prepare_price_panel(self.raw)
        weekend = self.raw.iloc[0].to_dict()
        weekend.update(Date=pd.Timestamp("2022-09-03"), Ticker=" ")
        invalid_date = self.raw.iloc[0].to_dict()
        invalid_date.update(Date="not-a-date", Ticker="CCC")
        malformed = pd.DataFrame([weekend, invalid_date])
        actual, audit = prepare_price_panel(pd.concat([self.raw, malformed], ignore_index=True))
        self.assertEqual(audit["dropped_invalid_keys"], 2)
        self.assertEqual(audit["grid_rows"], len(expected))
        assert_frame_equal(actual, expected)

    def test_normalization_and_metadata_validation(self):
        raw = self.raw.copy()
        raw["Ticker"] = " " + raw["Ticker"].str.lower() + " "
        raw["Volume"] = raw["Volume"].astype(str)
        meta = pd.DataFrame({"Ticker": ["aaa", "bbb"], "Sector": ["A", "B"]})
        panel, _ = prepare_price_panel(raw, meta)
        self.assertEqual(set(panel["Ticker"]), {"AAA", "BBB"})
        self.assertEqual(self.row(panel, "AAA", 70)["Sector"], "A")
        with self.assertRaisesRegex(ValueError, "one record per"):
            prepare_price_panel(raw, pd.concat([meta, pd.DataFrame({"Ticker": ["AAA"], "Sector": ["C"]})]))

    def test_signal_eligibility_does_not_require_future_labels(self):
        panel, _ = prepare_price_panel(self.raw)
        last = self.row(panel, "AAA", 99)
        self.assertTrue(last["SignalEligible"])
        self.assertTrue(pd.isna(last["target"]))
        damaged = self.raw.copy()
        damaged.loc[damaged["Date"].gt(self.dates[70]), "Volume"] = 0
        after, _ = prepare_price_panel(damaged)
        self.assertTrue(self.row(after, "AAA", 70)["SignalEligible"])
        self.assertTrue(pd.isna(self.row(after, "AAA", 70)["target"]))
        self.assertTrue(self.row(panel, "AAA", 70)["SignalEligible"])

    def test_shared_export_keeps_signals_without_future_labels(self):
        missing_future = self.raw["Ticker"].eq("AAA") & self.raw["Date"].eq(self.dates[71])
        panel, _ = prepare_price_panel(self.raw.loc[~missing_future])
        exported = make_model_dataset(panel, labeled_only=False)
        labeled = make_model_dataset(panel)
        for session in [70, 99]:
            row = self.row(exported, "AAA", session)
            self.assertTrue(np.isfinite(row[FEATURES].astype(float)).all())
            self.assertTrue(pd.isna(row["target"]))
            self.assertTrue(pd.isna(row["target_rank"]))
            self.assertFalse((labeled["Ticker"].eq("AAA") & labeled["Date"].eq(self.dates[session])).any())
        splits = add_date_splits(exported, self.dates[70], self.dates[85])
        self.assertEqual(self.row(splits, "AAA", 99)["Split"], "test")

    def test_future_price_changes_cannot_change_earlier_features(self):
        panel, _ = prepare_price_panel(self.raw)
        changed = self.raw.copy()
        columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        changed.loc[changed["Date"].gt(self.dates[70]), columns] *= 3
        after, _ = prepare_price_panel(changed)
        earlier = panel["Date"].le(self.dates[70])
        assert_frame_equal(panel.loc[earlier, FEATURES + ["SignalEligible"]], after.loc[earlier, FEATURES + ["SignalEligible"]])
        self.assertNotEqual(self.row(panel, "AAA", 70)["target"], self.row(after, "AAA", 70)["target"])

    def test_daily_target_rank_direction_and_ties(self):
        panel, _ = prepare_price_panel(self.raw)
        dataset = make_model_dataset(panel)
        date_rows = dataset.loc[dataset["Date"].eq(self.dates[70])].set_index("Ticker")
        self.assertEqual(date_rows.loc["AAA", "target_rank"], 1.0)
        self.assertEqual(date_rows.loc["BBB", "target_rank"], 0.5)
        self.assertEqual(date_rows.loc["AAA", "LabelEndDate"], self.dates[71])
        self.assertTrue(dataset["SignalEligible"].all())
        self.assertTrue(np.isfinite(dataset["target"]).all())
        panel.loc[panel["Date"].eq(self.dates[70]), "target"] = 0.1
        ties = make_model_dataset(panel)
        self.assertTrue(ties.loc[ties["Date"].eq(self.dates[70]), "target_rank"].eq(0.75).all())

    def test_purge_labels_that_cross_split_boundaries(self):
        panel, _ = prepare_price_panel(self.raw)
        result = add_date_splits(make_model_dataset(panel), self.dates[70], self.dates[85])
        for cutoff in [70, 85]:
            self.assertTrue(result.loc[result["Date"].eq(self.dates[cutoff]), "Split"].eq("purged").all())
        for split, cutoff in [("train", self.dates[70]), ("validation", self.dates[85])]:
            self.assertTrue(result.loc[result["Split"].eq(split), "LabelEndDate"].le(cutoff).all())
        self.assertTrue(result.loc[result["Date"].eq(self.dates[86]), "Split"].eq("test").all())
        self.assertTrue(result.groupby("Date")["Split"].nunique().eq(1).all())

    def test_boundary_purge_keeps_entire_signal_date_together(self):
        panel, _ = prepare_price_panel(self.raw)
        dataset = make_model_dataset(panel)
        boundary = dataset["Date"].eq(self.dates[69]) & dataset["Ticker"].eq("BBB")
        dataset.loc[boundary, "LabelEndDate"] = self.dates[72]
        result = add_date_splits(dataset, self.dates[71], self.dates[85])
        self.assertTrue(result.loc[result["Date"].eq(self.dates[69]), "Split"].eq("purged").all())


if __name__ == "__main__":
    unittest.main()
