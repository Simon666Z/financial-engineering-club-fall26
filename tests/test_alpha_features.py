"""Handcomputed alpha formulas, full-window gaps and causal invariants."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.alpha_features import ALPHA_FEATURES, ALPHA_FEATURE_METADATA, add_alpha_features


def fixture(sessions=540):
    dates = pd.bdate_range("2016-01-04", periods=sessions)
    rows = []
    for ticker, slope, adjustment in [("AAA", 0.002, 0.8), ("BBB", -0.001, 0.9)]:
        for index, date in enumerate(dates):
            price = 100 * np.exp(slope * index + 0.003 * np.sin(index / 3))
            previous = 100 * np.exp(slope * (index - 1) + 0.003 * np.sin((index - 1) / 3))
            close = price / adjustment
            intraday = 1 + 0.001 + 0.0005 * np.sin(index / 7)
            rows.append({
                "Date": date, "Ticker": ticker, "Adj Close": price,
                "Open": close / intraday, "Close": close,
                "High": close * 1.01, "Low": close * 0.99,
                "Volume": 1_000_000 * (1 + (index % 7) / 10),
                "ret_1d": price / previous - 1 if index else np.nan,
                "SignalEligible": bool(index >= 60),
                "target": 0.123, "target_rank": 0.5, "Split": "train",
                "LabelEndDate": dates[index + 1] if index + 1 < sessions else pd.NaT,
            })
    return pd.DataFrame(rows).sort_values(["Date", "Ticker"]).reset_index(drop=True), dates


class AlphaFeatureTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.dates = fixture()

    def stock(self, frame, ticker="AAA"):
        return frame.loc[frame["Ticker"].eq(ticker)].sort_values("Date").reset_index(drop=True)

    def test_momentum_and_high_match_endpoints_and_complete_windows(self):
        result = self.stock(add_alpha_features(self.panel))
        raw = self.stock(self.panel)
        session = 300
        p = raw["Adj Close"]
        self.assertAlmostEqual(result.loc[session, "mom_126_21"], p.iloc[session - 21] / p.iloc[session - 126] - 1)
        self.assertAlmostEqual(result.loc[session, "mom_252_21"], p.iloc[session - 21] / p.iloc[session - 252] - 1)
        self.assertAlmostEqual(result.loc[session, "high_ratio_252"], p.iloc[session] / p.iloc[session - 251:session + 1].max())
        self.assertTrue(result.loc[:125, "mom_126_21"].isna().all())
        self.assertTrue(pd.notna(result.loc[126, "mom_126_21"]))
        self.assertTrue(result.loc[:251, "mom_252_21"].isna().all())
        self.assertTrue(pd.notna(result.loc[252, "mom_252_21"]))
        self.assertTrue(result.loc[:250, "high_ratio_252"].isna().all())
        self.assertTrue(result.loc[:251, "AlphaSignalEligible"].eq(False).all())
        self.assertTrue(result.loc[252:, "AlphaSignalEligible"].all())

    def test_dollar_volume_percentile_includes_today_with_average_ties(self):
        # Exact integer-valued prices/volume make genuine ties, avoiding
        # rounding differences from reciprocal multiplication.
        self.panel["Close"] = 100.0
        self.panel["Volume"] = 10_000.0
        result = self.stock(add_alpha_features(self.panel))
        self.assertAlmostEqual(result.loc[300, "dollar_volume_rank_50"], 25.5 / 50)
        self.assertTrue(result.loc[:48, "dollar_volume_rank_50"].isna().all())
        self.assertTrue(pd.notna(result.loc[49, "dollar_volume_rank_50"]))
        # Give the last observation the only large volume in its window.
        mask = self.panel["Ticker"].eq("AAA") & self.panel["Date"].eq(self.dates[300])
        self.panel.loc[mask, "Volume"] *= 2
        result = self.stock(add_alpha_features(self.panel))
        self.assertEqual(result.loc[300, "dollar_volume_rank_50"], 1.0)

    def test_dollar_volume_shock_excludes_today_and_reversal_uses_positive_shock(self):
        self.panel["Volume"] = 1_000_000 / self.panel["Close"]
        mask = self.panel["Ticker"].eq("AAA") & self.panel["Date"].eq(self.dates[300])
        self.panel.loc[mask, "Volume"] *= 3
        result = self.stock(add_alpha_features(self.panel))
        expected = np.log(3)
        self.assertAlmostEqual(result.loc[300, "dollar_volume_log_shock_20"], expected)
        self.assertAlmostEqual(result.loc[300, "volume_reversal_1d"], -result.loc[300, "ret_1d"] * expected)
        self.assertNotAlmostEqual(expected, np.log(3 / 1.1))
        self.panel.loc[mask, "Volume"] /= 6
        result = self.stock(add_alpha_features(self.panel))
        self.assertLess(result.loc[300, "dollar_volume_log_shock_20"], 0)
        self.assertEqual(result.loc[300, "volume_reversal_1d"], 0.0)
        self.assertTrue(result.loc[:19, "dollar_volume_log_shock_20"].isna().all())

    def test_max_and_top_five_returns_match_hand_window(self):
        result = self.stock(add_alpha_features(self.panel))
        window = result.loc[280:300, "ret_1d"].to_numpy()
        self.assertEqual(len(window), 21)
        self.assertAlmostEqual(result.loc[300, "max_return_21d"], max(window))
        self.assertAlmostEqual(result.loc[300, "max5_return_21d"], np.sort(window)[-5:].mean())
        self.assertTrue(result.loc[:20, "max_return_21d"].isna().all())
        self.assertTrue(pd.notna(result.loc[21, "max_return_21d"]))

    def test_adjusted_overnight_and_intraday_decomposition_compounds(self):
        result = self.stock(add_alpha_features(self.panel))
        raw = self.stock(self.panel)
        session = 300
        expected_overnight = (raw.loc[session, "Adj Close"] / raw.loc[session - 1, "Adj Close"]) / (raw.loc[session, "Close"] / raw.loc[session, "Open"]) - 1
        self.assertAlmostEqual(result.loc[session, "overnight_ret_1d"], expected_overnight)
        for window in [20, 60]:
            observations = raw.iloc[session - window + 1:session + 1]
            intraday = observations["Close"].to_numpy() / observations["Open"].to_numpy()
            total = observations["Adj Close"].to_numpy() / raw["Adj Close"].iloc[session - window:session].to_numpy()
            overnight = total / intraday
            overnight_compound = np.prod(overnight) - 1
            intraday_compound = np.prod(intraday) - 1
            self.assertAlmostEqual(result.loc[session, f"overnight_mom_{window}d"], overnight_compound)
            self.assertAlmostEqual(result.loc[session, f"intraday_mom_{window}d"], intraday_compound)
            self.assertAlmostEqual(result.loc[session, f"overnight_intraday_spread_{window}d"], overnight_compound - intraday_compound)
            self.assertAlmostEqual((1 + overnight_compound) * (1 + intraday_compound), raw.loc[session, "Adj Close"] / raw.loc[session - window, "Adj Close"])

    def test_current_labels_future_labels_and_splits_never_change_features(self):
        before = add_alpha_features(self.panel)
        self.panel["target"] = np.nan
        self.panel["target_rank"] = np.inf
        self.panel["Split"] = "test"
        self.panel["LabelEndDate"] = pd.Timestamp("2100-01-01")
        after = add_alpha_features(self.panel)
        assert_frame_equal(before[ALPHA_FEATURES + ["AlphaSignalEligible"]], after[ALPHA_FEATURES + ["AlphaSignalEligible"]])

    def test_future_quotes_cannot_change_earlier_features(self):
        before = add_alpha_features(self.panel)
        cutoff = self.dates[400]
        suffix = self.panel["Date"].gt(cutoff)
        for name in ["Adj Close", "Open", "Close", "Volume"]:
            self.panel.loc[suffix, name] *= 5
        self.panel.loc[suffix, "ret_1d"] = 0.8
        after = add_alpha_features(self.panel)
        earlier = self.panel["Date"].le(cutoff)
        assert_frame_equal(before.loc[earlier, ALPHA_FEATURES + ["AlphaSignalEligible"]], after.loc[earlier, ALPHA_FEATURES + ["AlphaSignalEligible"]])

    def test_shuffled_rows_preserve_order_index_original_columns_and_same_features(self):
        expected = add_alpha_features(self.panel)
        shuffled = self.panel.sample(frac=1, random_state=7)
        result = add_alpha_features(shuffled)
        assert_frame_equal(result[self.panel.columns], shuffled)
        self.assertEqual(result.index.tolist(), shuffled.index.tolist())
        actual_sorted = result.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        assert_frame_equal(expected, actual_sorted)

    def test_ticker_histories_do_not_mix(self):
        before = add_alpha_features(self.panel)
        aaa = self.panel["Ticker"].eq("AAA")
        self.panel.loc[aaa, "Adj Close"] *= 2 + np.arange(aaa.sum()) / 100
        self.panel.loc[aaa, "Volume"] *= 10
        self.panel.loc[aaa, "ret_1d"] = 0.1
        after = add_alpha_features(self.panel)
        assert_frame_equal(self.stock(before, "BBB"), self.stock(after, "BBB"))
        self.assertNotAlmostEqual(self.stock(before).loc[300, "mom_126_21"], self.stock(after).loc[300, "mom_126_21"])

    def test_missing_quote_breaks_windows_and_requires_252_session_rewarmup(self):
        gap = self.panel["Ticker"].eq("AAA") & self.panel["Date"].eq(self.dates[270])
        self.panel.loc[gap, ["Adj Close", "Open", "Close", "Volume", "ret_1d"]] = np.nan
        self.panel.loc[gap, "SignalEligible"] = False
        result = self.stock(add_alpha_features(self.panel))
        self.assertTrue(result.loc[269, "AlphaSignalEligible"])
        self.assertFalse(result.loc[270:522, "AlphaSignalEligible"].any())
        self.assertTrue(result.loc[523, "AlphaSignalEligible"])
        # Even a gap in the skip-month region invalidates full momentum windows.
        self.assertTrue(pd.isna(result.loc[280, "mom_252_21"]))
        self.assertTrue(pd.isna(result.loc[280, "mom_126_21"]))
        self.assertTrue(pd.isna(result.loc[271, "overnight_ret_1d"]))
        self.assertTrue(pd.isna(result.loc[291, "max_return_21d"]))
        self.assertTrue(pd.notna(result.loc[292, "max_return_21d"]))

    def test_omitted_stock_session_is_not_bridged_or_inserted_in_output(self):
        gap = self.panel["Ticker"].eq("AAA") & self.panel["Date"].eq(self.dates[270])
        sparse = self.panel.loc[~gap].copy()
        result = add_alpha_features(sparse)
        self.assertEqual(len(result), len(sparse))
        self.assertEqual(result.index.tolist(), sparse.index.tolist())
        aaa = result.loc[result["Ticker"].eq("AAA")].set_index("Date")
        self.assertTrue(pd.isna(aaa.loc[self.dates[271], "overnight_ret_1d"]))
        self.assertFalse(aaa.loc[self.dates[271]:self.dates[522], "AlphaSignalEligible"].any())
        self.assertTrue(aaa.loc[self.dates[523], "AlphaSignalEligible"])

    def test_original_signal_eligibility_is_required_even_with_complete_new_features(self):
        mask = self.panel["Ticker"].eq("AAA") & self.panel["Date"].eq(self.dates[300])
        self.panel.loc[mask, "SignalEligible"] = False
        result = add_alpha_features(self.panel)
        self.assertTrue(np.isfinite(result.loc[mask, ALPHA_FEATURES].to_numpy()).all())
        self.assertFalse(result.loc[mask, "AlphaSignalEligible"].iloc[0])

    def test_nonpositive_nonfinite_inputs_are_missing_not_imputed(self):
        for column, bad in [("Adj Close", 0.0), ("Open", -1.0), ("Close", np.inf), ("Volume", -np.inf)]:
            panel = self.panel.copy()
            mask = panel["Ticker"].eq("AAA") & panel["Date"].eq(self.dates[300])
            panel.loc[mask, column] = bad
            with self.subTest(column=column):
                result = add_alpha_features(panel)
                self.assertFalse(result.loc[mask, "AlphaSignalEligible"].iloc[0])
                self.assertFalse(np.isinf(result[ALPHA_FEATURES].to_numpy()).any())
                self.assertEqual(result.loc[mask, column].iloc[0], bad)

    def test_repeated_normalized_keys_bad_keys_and_flag_strings_are_rejected(self):
        duplicate = self.panel.iloc[[0]].copy()
        duplicate["Ticker"] = " aaa "
        malformed = [
            (pd.concat([self.panel, duplicate]), "duplicate"),
            (self.panel.assign(Date="not-a-date"), "invalid Date/Ticker"),
            (self.panel.assign(Ticker=" "), "invalid Date/Ticker"),
            (self.panel.assign(SignalEligible="False"), "boolean"),
        ]
        for panel, message in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    add_alpha_features(panel)

    def test_missing_columns_empty_inputs_and_metadata(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            add_alpha_features(self.panel.drop(columns="Volume"))
        empty = self.panel.iloc[:0]
        result = add_alpha_features(empty)
        self.assertEqual(len(result), 0)
        self.assertEqual(len(ALPHA_FEATURES), 15)
        self.assertEqual(set(ALPHA_FEATURE_METADATA), set(ALPHA_FEATURES))
        self.assertTrue(set(ALPHA_FEATURES + ["AlphaSignalEligible"]).issubset(result))
        for feature in ALPHA_FEATURES:
            self.assertTrue(ALPHA_FEATURE_METADATA[feature]["adaptation"])
            self.assertTrue(ALPHA_FEATURE_METADATA[feature]["source_urls"])
            self.assertTrue(ALPHA_FEATURE_METADATA[feature]["formula"])


if __name__ == "__main__":
    unittest.main()
