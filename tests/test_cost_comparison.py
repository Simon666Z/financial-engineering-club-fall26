"""Transaction fees alter subsequent investable cash and compounding."""
import numpy as np
import pandas as pd
import pytest
from src.backtests.cost_comparison import evaluate_cost_scenario


def scenario(cost_bps):
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    panel = pd.DataFrame([{"Date": date, "Ticker": ticker, "ExecutionAverage": 10.0, "SignalEligible": True,
                           "MarkClose": 20.0 if ticker == "B" and date == dates[-1] else 10.0}
                          for date in dates for ticker in ["A", "B"]])
    scores = pd.DataFrame([{"Date": date, "Ticker": ticker, "score": (1.0 if ticker == ("A" if i == 0 else "B") else 0.0)}
                           for i, date in enumerate(dates) for ticker in ["A", "B"]])
    params = {"initial_capital": 1000.0, "entry_k": 1, "exit_k": 1, "long_fraction": 1.0,
              "cost_bps": 0.0, "allow_additions": True, "exit_unranked": True}
    result = evaluate_cost_scenario(panel, scores, prediction_column="score", parameters=params, cost_bps=cost_bps)
    assert params["cost_bps"] == 0.0
    return result


def test_costed_cash_is_reinvested_after_both_sides_pay_fees():
    evaluation, daily, trades, positions, fields = scenario(10.0)
    expected = 2000.0 * 0.999 / (1.001 ** 2)
    assert evaluation["final_net_equity"] == pytest.approx(expected)
    assert fields["cumulative_return_after_costs"] == pytest.approx(expected / 1000.0 - 1.0)
    assert fields["cost_adjusted_status"] == "resolved"
    assert evaluation["final_gross_equity"] == pytest.approx(2000.0)
    filled = trades[(trades.book == "net") & (trades.status == "filled")]
    assert list(filled.action) == ["entry", "exit", "entry"]
    np.testing.assert_allclose(filled.cost, filled.notional * 0.001)
    assert fields["transaction_costs_paid"] == pytest.approx(filled.cost.sum())
    assert expected != pytest.approx(2000.0 - fields["transaction_costs_paid"])
    assert daily.iloc[2].net_long_count == 0  # Exit proceeds wait until the following session.
    assert daily.iloc[-1].net_long_count == 1  # No invented terminal sale.


def test_zero_cost_scenario_reproduces_gross_result():
    evaluation, daily, trades, positions, fields = scenario(0.0)
    np.testing.assert_array_equal(daily.net_nav, daily.gross_nav)
    assert fields["cumulative_return_after_costs"] == pytest.approx(1.0)
    assert fields["transaction_costs_paid"] == 0.0
