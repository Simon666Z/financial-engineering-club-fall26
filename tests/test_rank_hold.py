"""Cash, short inventory, timing and causality tests with invented prices."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtests.rank_hold import evaluate_rank_hold


def fixture(sessions=6):
    dates = pd.bdate_range("2024-01-02", periods=sessions)
    rows = []
    for date in dates:
        for ticker in ["AAA", "BBB", "CCC", "DDD"]:
            rows.append({"Date": date, "Ticker": ticker, "SignalEligible": True,
                         "ExecutionAverage": 100.0, "MarkClose": 100.0})
    panel = pd.DataFrame(rows)
    predictions = panel[["Date", "Ticker"]].copy()
    predictions["prediction"] = predictions["Ticker"].map({"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0})
    return panel, predictions, dates


class RankHoldTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.predictions, self.dates = fixture()

    def run_book(self, **kwargs):
        defaults = dict(initial_capital=1_000_000, entry_k=1, exit_k=2, cost_bps=0)
        defaults.update(kwargs)
        return evaluate_rank_hold(self.panel, self.predictions, **defaults)

    def price(self, ticker, session, column, value):
        self.panel.loc[self.panel["Ticker"].eq(ticker) & self.panel["Date"].eq(self.dates[session]), column] = value

    def ranks(self, session, order):
        scores = {ticker: len(order) - i for i, ticker in enumerate(order)}
        mask = self.predictions["Date"].eq(self.dates[session])
        self.predictions.loc[mask, "prediction"] = self.predictions.loc[mask, "Ticker"].map(scores)

    def book_trades(self, trades, name="gross"):
        return trades.loc[trades["book"].eq(name)].copy()

    def test_initial_cash_first_fill_lag_and_no_forced_terminal_liquidation(self):
        summary, daily, trades, positions = self.run_book()
        self.assertEqual(daily.iloc[0]["gross_nav"], 1_000_000)
        self.assertEqual(daily.iloc[0]["gross_free_cash"], 1_000_000)
        self.assertEqual(daily.iloc[0]["gross_long_count"], 0)
        self.assertTrue(pd.isna(daily.iloc[0]["gross_return"]))
        self.assertEqual(daily.iloc[1]["gross_long_count"], 1)
        self.assertEqual(daily.iloc[1]["gross_short_count"], 1)
        filled = self.book_trades(trades).loc[trades["status"].eq("filled")]
        self.assertEqual(len(filled), 2)
        self.assertTrue(filled["SignalDate"].eq(self.dates[0]).all())
        self.assertTrue(filled["ExecutionDate"].eq(self.dates[1]).all())
        self.assertTrue(filled["action"].eq("entry").all())
        self.assertEqual(summary["first_execution_date"], "2024-01-03")
        self.assertEqual(summary["trading_dates"], 5)
        self.assertEqual(summary["net_final_long_count"], 1)
        self.assertEqual(summary["net_final_short_count"], 1)
        self.assertEqual(len(positions.loc[positions["Date"].eq(self.dates[-1])]), 4)

    def test_short_proceeds_are_collateral_not_reusable_cash(self):
        _, daily, trades, _ = self.run_book()
        row = daily.iloc[1]
        self.assertEqual(row["gross_free_cash"], 0.0)
        self.assertEqual(row["gross_short_collateral"], 500_000)
        self.assertEqual(row["gross_short_proceeds"], 500_000)
        self.assertEqual(row["gross_short_liability"], 500_000)
        self.assertEqual(row["gross_long_value"], 500_000)
        self.assertEqual(row["gross_nav"], 1_000_000)
        self.assertEqual(row["gross_planned_entry_budget"], 0.0)
        self.assertEqual(row["gross_gross_exposure"], 1.0)
        self.assertEqual(row["gross_net_exposure"], 0.0)
        self.assertEqual(len(self.book_trades(trades)), 2)

    def test_symmetric_long_gain_short_decline_and_close_nav(self):
        self.price("AAA", 1, "MarkClose", 110.0)
        self.price("DDD", 1, "MarkClose", 90.0)
        summary, daily, _, positions = self.run_book()
        row = daily.iloc[1]
        self.assertEqual(row["gross_long_value"], 550_000)
        self.assertEqual(row["gross_short_liability"], 450_000)
        self.assertEqual(row["gross_nav"], 1_100_000)
        self.assertAlmostEqual(row["gross_return"], 0.1)
        self.assertAlmostEqual(row["gross_net_exposure"], 100_000 / 1_100_000)
        first = positions.loc[positions["Date"].eq(self.dates[1]) & positions["book"].eq("gross")]
        self.assertEqual(first["unrealized_pnl_before_fees"].sum(), 100_000)
        self.assertEqual(summary["initial_capital"], 1_000_000)

    def test_exact_rank_threshold_and_next_session_exit(self):
        self.ranks(1, ["BBB", "AAA", "DDD", "CCC"])
        self.ranks(2, ["BBB", "CCC", "AAA", "DDD"])
        _, daily, trades, positions = self.run_book()
        first = self.book_trades(trades)
        exits = first.loc[first["action"].eq("exit") & first["Ticker"].eq("AAA")]
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits.iloc[0]["rank"], 3)
        self.assertEqual(exits.iloc[0]["SignalDate"], self.dates[2])
        self.assertEqual(exits.iloc[0]["ExecutionDate"], self.dates[3])
        held_at_signal = positions.loc[positions["book"].eq("gross") & positions["Date"].eq(self.dates[2]) & positions["Ticker"].eq("AAA")]
        self.assertEqual(len(held_at_signal), 1)
        self.assertEqual(daily.iloc[2]["gross_long_count"], 1)

    def test_exit_cash_funds_following_signal_not_same_session_entries(self):
        self.ranks(1, ["DDD", "BBB", "CCC", "AAA"])
        self.ranks(2, ["DDD", "BBB", "CCC", "AAA"])
        _, daily, trades, _ = self.run_book()
        gross = self.book_trades(trades)
        on_exit_day = gross.loc[gross["ExecutionDate"].eq(self.dates[2])]
        self.assertTrue(on_exit_day["action"].eq("exit").all())
        self.assertEqual(daily.iloc[1]["gross_planned_entry_budget"], 0.0)
        self.assertEqual(daily.iloc[2]["gross_free_cash"], 1_000_000)
        self.assertEqual(daily.iloc[2]["gross_planned_entry_budget"], 1_000_000)
        new_entries = gross.loc[gross["ExecutionDate"].eq(self.dates[3]) & gross["action"].eq("entry")]
        self.assertEqual(len(new_entries), 2)
        self.assertTrue(new_entries["SignalDate"].eq(self.dates[2]).all())

    def test_retained_positions_can_accumulate_and_receive_additions(self):
        self.ranks(1, ["BBB", "CCC", "DDD", "AAA"])
        self.ranks(2, ["BBB", "CCC", "DDD", "AAA"])
        self.ranks(3, ["BBB", "CCC", "DDD", "AAA"])
        _, daily, _, positions = self.run_book()
        self.assertEqual(daily.iloc[3]["gross_long_count"], 1)
        self.assertEqual(daily.iloc[3]["gross_short_count"], 2)
        self.assertGreater(daily.iloc[3]["gross_long_count"] + daily.iloc[3]["gross_short_count"], 2)
        self.price("AAA", 1, "ExecutionAverage", np.nan)
        # Half the first allocation stays cash; the retained short can be topped up.
        self.predictions["prediction"] = self.predictions["Ticker"].map({"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0})
        _, _, _, positions = self.run_book()
        ddd = positions.loc[positions["book"].eq("gross") & positions["Ticker"].eq("DDD")]
        q1 = ddd.loc[ddd["Date"].eq(self.dates[1]), "quantity"].iloc[0]
        q2 = ddd.loc[ddd["Date"].eq(self.dates[2]), "quantity"].iloc[0]
        self.assertEqual(q1, 5_000)
        self.assertEqual(q2, 7_500)

    def test_entries_use_prior_cash_and_capital_funds_commission(self):
        summary, daily, trades, _ = self.run_book(cost_bps=10)
        net = self.book_trades(trades, "net")
        filled = net.loc[net["status"].eq("filled")]
        self.assertEqual(len(filled), 2)
        np.testing.assert_allclose(filled["planned_budget"], [500_000, 500_000])
        np.testing.assert_allclose(filled["notional"] + filled["cost"], filled["funded_budget"])
        self.assertAlmostEqual(daily.iloc[1]["net_free_cash"], 0.0)
        self.assertAlmostEqual(daily.iloc[1]["net_nav"], 1_000_000 / 1.001)
        self.assertAlmostEqual(summary["net_transaction_costs"], 1_000_000 * 0.001 / 1.001)
        self.assertEqual(summary["gross_transaction_costs"], 0.0)
        self.assertGreater(daily.iloc[1]["gross_long_value"], daily.iloc[1]["net_long_value"])

    def test_short_cover_and_long_sale_pay_second_side_commission(self):
        for session in [1, 2, 3, 4, 5]:
            self.ranks(session, ["DDD", "BBB", "CCC", "AAA"])
        _, daily, trades, _ = self.run_book(cost_bps=10)
        row = daily.iloc[2]
        expected = 1_000_000 * 0.999 / 1.001
        self.assertAlmostEqual(row["net_nav"], expected)
        self.assertAlmostEqual(row["net_free_cash"], expected)
        self.assertEqual(row["net_long_count"], 0)
        self.assertEqual(row["net_short_count"], 0)
        self.assertAlmostEqual(row["net_cumulative_transaction_cost"], 1_000_000 - expected)
        exits = self.book_trades(trades, "net").loc[trades["action"].eq("exit")]
        self.assertTrue(exits["cost"].gt(0).all())
        self.assertEqual(set(exits["side"]), {"long", "short"})

    def test_missing_entry_keeps_original_slot_cash_without_replacement(self):
        self.price("AAA", 1, "ExecutionAverage", np.nan)
        summary, daily, trades, positions = self.run_book()
        row = daily.iloc[1]
        self.assertEqual(row["gross_free_cash"], 500_000)
        self.assertEqual(row["gross_long_count"], 0)
        first = self.book_trades(trades).loc[trades["ExecutionDate"].eq(self.dates[1])]
        self.assertEqual(set(first["Ticker"]), {"AAA", "DDD"})
        self.assertEqual(first.loc[first["Ticker"].eq("AAA"), "status"].iloc[0], "missing_entry_fill")
        self.assertGreater(summary["missing_entry_orders"], 0)
        self.assertFalse(positions.loc[positions["Date"].eq(self.dates[1]), "Ticker"].isin(["BBB", "CCC"]).any())

    def test_missing_exit_retains_inventory_and_retries_without_false_sale(self):
        for session in [1, 2, 3, 4, 5]:
            self.ranks(session, ["DDD", "BBB", "CCC", "AAA"])
        self.price("AAA", 2, "ExecutionAverage", np.nan)
        summary, _, trades, positions = self.run_book()
        missing = self.book_trades(trades).loc[trades["status"].eq("missing_exit_fill")]
        self.assertEqual(missing.iloc[0]["Ticker"], "AAA")
        self.assertEqual(missing.iloc[0]["notional"], 0.0)
        retained = positions.loc[positions["book"].eq("gross") & positions["Date"].eq(self.dates[2]) & positions["Ticker"].eq("AAA")]
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained.iloc[0]["quantity"], 5_000)
        self.assertEqual(summary["missing_exit_orders"], 2)

    def test_missing_held_mark_fails_with_ticker_and_date(self):
        self.price("DDD", 2, "MarkClose", np.nan)
        with self.assertRaisesRegex(ValueError, "Unresolved held mark.*DDD.*2024-01-04"):
            self.run_book()

    def test_unranked_known_at_signal_closes_next_session(self):
        removed = self.panel["Ticker"].eq("AAA") & self.panel["Date"].eq(self.dates[1])
        self.panel.loc[removed, "SignalEligible"] = False
        self.predictions = self.predictions.loc[~(self.predictions["Ticker"].eq("AAA") & self.predictions["Date"].eq(self.dates[1]))]
        _, _, trades, _ = self.run_book()
        exit_aaa = self.book_trades(trades).loc[trades["action"].eq("exit") & trades["Ticker"].eq("AAA")]
        self.assertEqual(exit_aaa.iloc[0]["SignalDate"], self.dates[1])
        self.assertEqual(exit_aaa.iloc[0]["ExecutionDate"], self.dates[2])
        self.assertTrue(pd.isna(exit_aaa.iloc[0]["rank"]))

    def test_future_fill_prices_do_not_change_frozen_dollar_budgets(self):
        _, _, before, _ = self.run_book(cost_bps=10)
        self.price("AAA", 1, "ExecutionAverage", 200.0)
        _, _, after, _ = self.run_book(cost_bps=10)
        a = self.book_trades(before, "net").loc[before["SignalDate"].eq(self.dates[0])].reset_index(drop=True)
        b = self.book_trades(after, "net").loc[after["SignalDate"].eq(self.dates[0])].reset_index(drop=True)
        assert_frame_equal(a[["Ticker", "side", "planned_budget", "funded_budget", "notional", "cost"]], b[["Ticker", "side", "planned_budget", "funded_budget", "notional", "cost"]])
        self.assertAlmostEqual(a.loc[a["Ticker"].eq("AAA"), "quantity"].iloc[0], 2 * b.loc[b["Ticker"].eq("AAA"), "quantity"].iloc[0])

    def test_future_suffix_prices_and_scores_cannot_change_earlier_path(self):
        _, daily_a, trades_a, positions_a = self.run_book()
        suffix = self.panel["Date"].ge(self.dates[4])
        self.panel.loc[suffix, "ExecutionAverage"] *= 3
        self.panel.loc[suffix, "MarkClose"] *= 2
        self.ranks(4, ["DDD", "CCC", "BBB", "AAA"])
        self.ranks(5, ["CCC", "DDD", "AAA", "BBB"])
        _, daily_b, trades_b, positions_b = self.run_book()
        assert_frame_equal(daily_a.loc[daily_a["Date"].lt(self.dates[4])].reset_index(drop=True), daily_b.loc[daily_b["Date"].lt(self.dates[4])].reset_index(drop=True))
        assert_frame_equal(trades_a.loc[trades_a["ExecutionDate"].lt(self.dates[4])].reset_index(drop=True), trades_b.loc[trades_b["ExecutionDate"].lt(self.dates[4])].reset_index(drop=True))
        assert_frame_equal(positions_a.loc[positions_a["Date"].lt(self.dates[4])].reset_index(drop=True), positions_b.loc[positions_b["Date"].lt(self.dates[4])].reset_index(drop=True))

    def test_per_security_adjusted_unit_rescaling_preserves_cash_and_nav(self):
        self.price("AAA", 1, "MarkClose", 110.0)
        self.price("DDD", 1, "MarkClose", 90.0)
        _, original, _, original_positions = self.run_book(cost_bps=10)
        factor = self.panel["Ticker"].map({"AAA": 2.0, "BBB": 3.0, "CCC": 4.0, "DDD": 5.0})
        self.panel["ExecutionAverage"] *= factor
        self.panel["MarkClose"] *= factor
        _, scaled, _, scaled_positions = self.run_book(cost_bps=10)
        np.testing.assert_allclose(original["net_nav"], scaled["net_nav"])
        np.testing.assert_allclose(original["gross_nav"], scaled["gross_nav"])
        np.testing.assert_allclose(original["net_free_cash"], scaled["net_free_cash"])
        q1 = original_positions.loc[original_positions["Ticker"].eq("AAA"), "quantity"].to_numpy()
        q2 = scaled_positions.loc[scaled_positions["Ticker"].eq("AAA"), "quantity"].to_numpy()
        np.testing.assert_allclose(q1, 2 * q2)

    def test_nav_and_returns_compound_without_resetting_initial_capital(self):
        self.price("AAA", 1, "MarkClose", 120.0)
        self.price("DDD", 1, "MarkClose", 80.0)
        self.price("AAA", 2, "MarkClose", 130.0)
        self.price("DDD", 2, "MarkClose", 70.0)
        summary, daily, _, _ = self.run_book()
        self.assertAlmostEqual(daily.iloc[1]["gross_nav"], 1_200_000)
        self.assertAlmostEqual(daily.iloc[2]["gross_nav"], 1_300_000)
        self.assertAlmostEqual(daily.iloc[2]["gross_return"], 1_300_000 / 1_200_000 - 1)
        growth = np.prod(1 + daily["gross_return"].iloc[1:])
        self.assertAlmostEqual(growth, summary["final_gross_equity"] / 1_000_000)

    def test_extreme_short_cover_debt_is_visible_and_blocks_new_risk(self):
        for session in [1, 2, 3, 4, 5]:
            self.ranks(session, ["DDD", "BBB", "CCC", "AAA"])
        self.price("DDD", 2, "ExecutionAverage", 400.0)
        summary, daily, trades, _ = self.run_book()
        self.assertEqual(daily.iloc[2]["gross_free_cash"], 0.0)
        self.assertEqual(daily.iloc[2]["gross_margin_debt"], 500_000)
        self.assertEqual(daily.iloc[2]["gross_nav"], -500_000)
        self.assertEqual(daily.iloc[2]["gross_status"], "insolvent")
        self.assertTrue(summary["gross_insolvency_observed"])
        self.assertGreater(summary["gross_margin_shortfall_events"], 0)
        later = self.book_trades(trades).loc[trades["ExecutionDate"].gt(self.dates[2])]
        self.assertFalse(later["action"].eq("entry").any())
        self.assertFalse(summary["gross_return_path_defined"])

    def test_bankruptcy_accounting_recovery_does_not_restore_performance(self):
        self.ranks(1, ["DDD", "AAA", "BBB", "CCC"])
        self.price("DDD", 2, "ExecutionAverage", 400.0)
        for session in [3, 4, 5]:
            self.price("AAA", session, "MarkClose", 400.0)
        summary, daily, trades, _ = self.run_book()
        self.assertEqual(daily.iloc[2]["gross_nav"], -500_000)
        self.assertEqual(daily.iloc[-1]["gross_nav"], 1_000_000)
        self.assertEqual(daily.iloc[-1]["gross_status"], "post_insolvency")
        self.assertFalse(summary["gross_performance_valid"])
        self.assertFalse(summary["performance_valid"])
        self.assertFalse(summary["fully_resolved"])
        self.assertTrue(summary["quote_resolution_complete"])
        self.assertTrue(pd.isna(summary["gross_cumulative_return"]))
        self.assertTrue(pd.isna(summary["gross_annualized_sharpe"]))
        self.assertTrue(pd.isna(summary["gross_max_drawdown"]))
        self.assertEqual(summary["final_gross_equity"], 1_000_000)
        later = self.book_trades(trades).loc[trades["ExecutionDate"].gt(self.dates[2])]
        self.assertFalse(later["action"].eq("entry").any())

    def test_external_funding_without_bankruptcy_also_invalidates_metrics(self):
        self.ranks(1, ["DDD", "AAA", "BBB", "CCC"])
        self.price("DDD", 2, "ExecutionAverage", 300.0)
        for session in [2, 3, 4, 5]:
            self.price("AAA", session, "MarkClose", 200.0)
        summary, daily, _, _ = self.run_book()
        self.assertEqual(daily.iloc[2]["gross_nav"], 500_000)
        self.assertEqual(daily.iloc[2]["gross_margin_debt"], 500_000)
        self.assertFalse(summary["gross_insolvency_observed"])
        self.assertFalse(summary["funding_valid"])
        self.assertFalse(summary["gross_performance_valid"])
        self.assertTrue(pd.isna(summary["net_cumulative_return"]))
        self.assertTrue(pd.isna(summary["annualized_sharpe"]))
        self.assertTrue(pd.isna(summary["max_drawdown"]))

    def test_complete_prediction_coverage_and_duplicates_are_rejected(self):
        for bad, message in [
            (self.predictions.iloc[1:], "coverage mismatch"),
            (self.predictions.loc[self.predictions["Date"].lt(self.dates[-1])], "coverage mismatch"),
            (pd.concat([self.predictions, self.predictions.iloc[[0]]]), "duplicate"),
            (self.predictions.assign(prediction=np.inf), "finite numeric"),
        ]:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_rank_hold(self.panel, bad, entry_k=1, exit_k=2)

    def test_targets_and_other_future_fields_are_ignored(self):
        summary_a, daily_a, trades_a, positions_a = self.run_book()
        self.panel["target"] = np.inf
        self.panel["LabelEndDate"] = pd.Timestamp("2100-01-01")
        self.panel["target_rank"] = -999
        summary_b, daily_b, trades_b, positions_b = self.run_book()
        self.assertEqual(summary_a, summary_b)
        assert_frame_equal(daily_a, daily_b)
        assert_frame_equal(trades_a, trades_b)
        assert_frame_equal(positions_a, positions_b)

    def test_small_universe_and_ties_have_disjoint_stable_tails(self):
        self.predictions["prediction"] = 0.0
        self.predictions = self.predictions.sample(frac=1, random_state=11)
        _, _, trades, _ = self.run_book(entry_k=20, exit_k=100)
        first = self.book_trades(trades).loc[trades["ExecutionDate"].eq(self.dates[1])]
        self.assertEqual(first.loc[first["side"].eq("long"), "Ticker"].tolist(), ["AAA", "BBB"])
        self.assertEqual(first.loc[first["side"].eq("short"), "Ticker"].tolist(), ["DDD", "CCC"])
        self.assertEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
