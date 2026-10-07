"""Handcomputed tests of execution timing, missing quotes, costs and coverage."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtests.showcase import evaluate_predictions, score_predictions


def fixture():
    dates = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for i, date in enumerate(dates):
        for ticker, target in [("AAA", -0.2), ("BBB", 0.2), ("CCC", 0.0)]:
            rows.append({
                "Date": date, "Ticker": ticker, "QuoteValid": True,
                "SignalEligible": True, "Open": 100.0, "Close": 100.0,
                "Adj Close": 100.0, "target": target if i < 3 else np.nan,
                "LabelEndDate": dates[i + 1] if i < 3 else pd.NaT,
            })
    frame = pd.DataFrame(rows)
    predictions = frame[["Date", "Ticker"]].copy()
    predictions["prediction"] = predictions["Ticker"].map({"AAA": -1.0, "BBB": 1.0, "CCC": 0.0})
    return frame, predictions, dates


class ShowcaseBacktestTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.predictions, self.dates = fixture()

    def price(self, ticker, session, column, value):
        mask = self.frame["Ticker"].eq(ticker) & self.frame["Date"].eq(self.dates[session])
        self.frame.loc[mask, column] = value

    def first_predictions(self, ticker):
        self.predictions["prediction"] = self.predictions["Ticker"].eq(ticker).astype(float)

    def only_eligible(self, tickers):
        self.frame["SignalEligible"] = self.frame["Ticker"].isin(tickers)
        self.predictions = self.predictions.loc[self.predictions["Ticker"].isin(tickers)].copy()

    def test_rank_target_is_not_execution_profit_or_eligibility(self):
        # BBB has the best close-to-close label but loses 10% during execution.
        self.price("BBB", 1, "Close", 90.0)
        summary, daily, trades = evaluate_predictions(self.frame, self.predictions, top_k=1, cost_bps=0)
        self.assertAlmostEqual(summary["rank_ic_mean"], 1.0)
        self.assertAlmostEqual(daily.iloc[0]["gross_return"], -0.1)
        self.assertAlmostEqual(summary["max_drawdown"], -0.1)
        self.assertEqual(trades.loc[trades["portfolio"].eq("strategy"), "Ticker"].tolist(), ["BBB"] * 3)
        self.frame.loc[self.frame["Ticker"].eq("BBB"), "target"] = np.nan
        changed, changed_daily, changed_trades = evaluate_predictions(self.frame, self.predictions, top_k=1, cost_bps=0)
        np.testing.assert_allclose(daily["net_return"], changed_daily["net_return"])
        self.assertEqual(trades["Ticker"].tolist(), changed_trades["Ticker"].tolist())
        self.assertEqual(changed["prediction_rows"], summary["prediction_rows"])

    def test_missing_selected_entry_keeps_cash_without_replacement_or_date_bridge(self):
        self.first_predictions("AAA")
        self.price("AAA", 1, "Open", np.nan)
        self.price("AAA", 2, "Close", 150.0)
        self.price("BBB", 1, "Close", 110.0)
        summary, daily, trades = evaluate_predictions(self.frame, self.predictions, top_k=1, cost_bps=0)
        self.assertEqual(daily.iloc[0]["executed_count"], 0)
        self.assertEqual(daily.iloc[0]["missing_entry_count"], 1)
        self.assertEqual(daily.iloc[0]["cash_weight"], 1.0)
        self.assertEqual(daily.iloc[0]["net_return"], 0.0)
        self.assertAlmostEqual(daily.iloc[0]["benchmark_net_return"], 0.1 / 3)
        first = trades.loc[trades["portfolio"].eq("strategy") & trades["SignalDate"].eq(self.dates[0])].iloc[0]
        self.assertEqual(first["Ticker"], "AAA")
        self.assertEqual(first["HoldingDate"], self.dates[1])
        self.assertEqual(first["status"], "missing_entry")
        self.assertEqual(summary["missing_entry_slots"], 1)

    def test_missing_exit_is_unresolved_and_excluded_with_a_count(self):
        self.first_predictions("AAA")
        self.price("AAA", 1, "Close", np.nan)
        summary, daily, trades = evaluate_predictions(self.frame, self.predictions, top_k=1, cost_bps=0)
        self.assertTrue(daily.iloc[0]["unresolved"])
        self.assertTrue(pd.isna(daily.iloc[0]["net_return"]))
        self.assertEqual(summary["unresolved_trading_dates"], 1)
        self.assertEqual(summary["aggregate_excluded_unresolved_dates"], 1)
        self.assertEqual(summary["aggregate_resolved_dates"], 2)
        self.assertFalse(summary["fully_resolved"])
        self.assertIn("conditional", summary["aggregate_scope"])
        unknown = trades.loc[trades["status"].eq("unresolved_exit")]
        self.assertEqual(len(unknown), 2)  # One trade in each portfolio.
        self.assertTrue(unknown["return_contribution"].isna().all())

    def test_preserved_execution_prices_detect_cleaner_masked_exit(self):
        self.first_predictions("AAA")
        self.frame["ExecutionOpen"] = self.frame["Open"]
        self.frame["ExecutionClose"] = self.frame["Close"]
        self.price("AAA", 1, "ExecutionClose", np.nan)
        self.price("AAA", 1, "Open", np.nan)
        self.price("AAA", 1, "Close", np.nan)
        self.price("AAA", 1, "QuoteValid", False)
        summary, daily, _ = evaluate_predictions(self.frame, self.predictions, top_k=1)
        self.assertEqual(daily.iloc[0]["executed_count"], 1)
        self.assertEqual(daily.iloc[0]["missing_entry_count"], 0)
        self.assertTrue(daily.iloc[0]["unresolved"])
        self.assertEqual(summary["unresolved_trading_dates"], 1)

    def test_benchmark_only_unresolved_uses_same_aggregate_dates(self):
        self.first_predictions("BBB")
        self.price("AAA", 1, "Close", np.nan)
        self.price("BBB", 1, "Close", 140.0)
        summary, daily, _ = evaluate_predictions(self.frame, self.predictions, top_k=1, cost_bps=0)
        self.assertAlmostEqual(daily.iloc[0]["net_return"], 0.4)
        self.assertFalse(daily.iloc[0]["unresolved"])
        self.assertTrue(daily.iloc[0]["benchmark_unresolved"])
        self.assertEqual(summary["unresolved_trading_dates"], 0)
        self.assertEqual(summary["benchmark_unresolved_trading_dates"], 1)
        self.assertEqual(summary["aggregate_resolved_dates"], 2)
        self.assertAlmostEqual(summary["net_cumulative_return"], 0.0)

    def test_ticker_ties_are_stable_and_ic_is_undefined(self):
        self.predictions["prediction"] = 0.0
        self.predictions = self.predictions.sample(frac=1, random_state=10)
        summary, daily, trades = evaluate_predictions(self.frame, self.predictions, top_k=1, cost_bps=0)
        strategy = trades.loc[trades["portfolio"].eq("strategy")]
        self.assertEqual(strategy["Ticker"].tolist(), ["AAA"] * 3)
        self.assertEqual(summary["rank_ic_scored_dates"], 0)
        self.assertEqual(summary["rank_ic_undefined_dates"], 4)
        self.assertTrue(pd.isna(summary["rank_ic_mean"]))
        self.assertTrue(daily["rank_ic"].isna().all())

    def test_costs_include_both_sides_and_never_overspend_capital(self):
        self.only_eligible(["AAA"])
        self.price("AAA", 1, "Close", 110.0)
        summary, daily, trades = evaluate_predictions(self.frame, self.predictions, top_k=2, cost_bps=10)
        row = daily.iloc[0]
        expected = 0.5 * (1.1 * 0.999 / 1.001 - 1)
        self.assertAlmostEqual(row["gross_return"], 0.05)
        self.assertAlmostEqual(row["net_return"], expected)
        self.assertAlmostEqual(row["benchmark_net_return"], 2 * expected)
        self.assertAlmostEqual(row["cash_weight"], 0.5)
        first = trades.loc[trades["portfolio"].eq("strategy")].iloc[0]
        self.assertAlmostEqual(first["buy_notional"] + first["buy_cost"], 0.5)
        self.assertAlmostEqual(first["buy_cost"], 0.001 * 0.5 / 1.001)
        self.assertAlmostEqual(first["sell_cost"], 0.001 * 0.5 * 1.1 / 1.001)
        self.assertAlmostEqual(first["sell_notional"] - first["sell_cost"] + 0.5, 1 + expected)
        self.assertAlmostEqual(row["turnover"], 0.5 * 2.1 / 1.001)
        self.assertAlmostEqual(summary["net_cumulative_return"], (1 + expected) * (0.999 / 1.001 * 0.5 + 0.5) ** 2 - 1)

    def test_missing_slots_have_no_cost_and_benchmark_uses_same_timing(self):
        self.only_eligible(["AAA"])
        self.price("AAA", 1, "Open", np.nan)
        _, daily, _ = evaluate_predictions(self.frame, self.predictions, top_k=1, cost_bps=10)
        self.assertEqual(daily.iloc[0]["net_return"], 0.0)
        self.assertEqual(daily.iloc[0]["benchmark_net_return"], 0.0)
        self.assertEqual(daily.iloc[0]["turnover"], 0.0)
        self.assertEqual(daily.iloc[0]["benchmark_turnover"], 0.0)
        self.assertAlmostEqual(daily.iloc[1]["net_return"], 0.999 / 1.001 - 1)

    def test_final_date_predictions_are_required_but_not_traded(self):
        summary, daily, trades = evaluate_predictions(self.frame, self.predictions)
        self.assertEqual(summary["prediction_rows"], 12)
        self.assertEqual(summary["excluded_final_signal_dates"], 1)
        self.assertEqual(summary["excluded_final_prediction_rows"], 3)
        self.assertEqual(summary["labeled_prediction_rows"], 9)
        self.assertEqual(len(daily), 3)
        self.assertEqual(daily["HoldingDate"].max(), self.dates[-1])
        self.assertLess(trades["SignalDate"].max(), self.dates[-1])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            evaluate_predictions(self.frame, self.predictions.loc[self.predictions["Date"].lt(self.dates[-1])])

    def test_label_beyond_truncated_snapshot_is_not_scored(self):
        shortened = self.frame.loc[self.frame["Date"].le(self.dates[2])]
        submitted = self.predictions.loc[self.predictions["Date"].le(self.dates[2])]
        summary, scores = score_predictions(shortened, submitted)
        self.assertEqual(summary["labeled_prediction_rows"], 6)
        self.assertEqual(summary["rank_ic_scored_dates"], 2)
        self.assertTrue(pd.isna(scores.iloc[-1]["rank_ic"]))

    def test_empty_signal_day_is_explicit_cash(self):
        self.frame.loc[self.frame["Date"].eq(self.dates[0]), "SignalEligible"] = False
        self.predictions = self.predictions.loc[self.predictions["Date"].ne(self.dates[0])]
        summary, daily, _ = evaluate_predictions(self.frame, self.predictions)
        self.assertEqual(daily.iloc[0]["selected_count"], 0)
        self.assertEqual(daily.iloc[0]["cash_weight"], 1.0)
        self.assertEqual(daily.iloc[0]["benchmark_cash_weight"], 1.0)
        self.assertEqual(daily.iloc[0]["net_return"], 0.0)
        self.assertEqual(summary["calendar_test_dates"], 4)
        self.assertEqual(summary["signal_dates"], 3)

    def test_coverage_duplicates_nonfinite_and_unknown_signals_rejected(self):
        invalid_tables = [
            (self.predictions.iloc[1:], "coverage mismatch"),
            (pd.concat([self.predictions, self.predictions.iloc[[0]]]), "duplicate"),
            (self.predictions.assign(prediction=np.nan), "finite numeric"),
            (self.predictions.assign(prediction=np.inf), "finite numeric"),
            (pd.concat([self.predictions, self.predictions.iloc[[0]].assign(Ticker="UNKNOWN")]), "coverage mismatch"),
            (pd.concat([self.predictions, self.predictions.iloc[[0]].assign(Date="2023-12-29")]), "coverage mismatch"),
        ]
        for submitted, message in invalid_tables:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_predictions(self.frame, submitted)
        self.frame.loc[0, "SignalEligible"] = False
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            evaluate_predictions(self.frame, self.predictions)

    def test_start_boundary_preserves_shared_calendar(self):
        start = self.dates[1]
        submitted = self.predictions.loc[self.predictions["Date"].ge(start)]
        summary, daily, _ = evaluate_predictions(self.frame, submitted, start_date=start)
        self.assertEqual(summary["start_date"], "2024-01-03")
        self.assertEqual(daily.iloc[0]["SignalDate"], self.dates[1])
        self.assertEqual(daily.iloc[0]["HoldingDate"], self.dates[2])


if __name__ == "__main__":
    unittest.main()
