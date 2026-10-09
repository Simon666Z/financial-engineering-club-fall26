"""Replay frozen rankings with independently funded transaction costs."""
from __future__ import annotations

import numpy as np
from src.backtests.rank_hold import evaluate_rank_hold


def evaluate_cost_scenario(panel, forecasts, *, prediction_column, parameters, cost_bps):
    """Return costed ledgers and report fields without changing the base book."""
    settings = {**parameters, "cost_bps": cost_bps}
    try:
        evaluation, daily, trades, positions = evaluate_rank_hold(
            panel, forecasts, prediction_column=prediction_column, **settings)
    except ValueError as error:
        if not str(error).startswith("Unresolved held mark:"):
            raise
        evaluation = {"status": "unresolved", "failure": str(error),
                      "cost_bps": float(cost_bps), "net_performance_valid": False,
                      "fully_resolved": False, "net_cumulative_return": np.nan,
                      "final_net_equity": np.nan, "net_transaction_costs": np.nan}
        daily = trades = positions = None
    valid = bool(evaluation.get("net_performance_valid", False))
    finite = all(np.isfinite(evaluation.get(name, np.nan)) for name in
                 ("net_cumulative_return", "final_net_equity"))
    valid = valid and finite
    status = "resolved" if valid else evaluation.get("status", "invalid")
    reason = "" if valid else (evaluation.get("failure") or evaluation.get("performance_invalidation_reason")
                               or "The cost-adjusted portfolio has no valid performance estimate.")
    evaluation.update(status=status, failure=reason)
    fields = {"cumulative_return_after_costs": evaluation["net_cumulative_return"] if valid else np.nan,
              "transaction_cost_bps": float(cost_bps),
              "final_equity_after_costs": evaluation["final_net_equity"] if valid else np.nan,
              "transaction_costs_paid": evaluation.get("net_transaction_costs", np.nan),
              "cost_adjusted_status": status, "cost_adjusted_failure": reason}
    return evaluation, daily, trades, positions, fields
