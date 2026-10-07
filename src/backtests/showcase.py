"""Chronological forecast scoring and a capital-accounted teaching backtest.

Signals use completed day-t features. Both portfolios buy the following shared
calendar session's open and sell its close. Forecast targets are used ONLY for
rank IC, never for stock selection or trading returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DAILY_COLUMNS = [
    "SignalDate", "HoldingDate", "gross_return", "net_return", "turnover",
    "unresolved", "selected_count", "executed_count", "missing_entry_count",
    "cash_weight", "rank_ic", "benchmark_gross_return", "benchmark_net_return",
    "benchmark_turnover", "benchmark_unresolved", "benchmark_selected_count",
    "benchmark_executed_count", "benchmark_missing_entry_count", "benchmark_cash_weight",
]
_TRADE_COLUMNS = [
    "portfolio", "SignalDate", "HoldingDate", "Ticker", "prediction",
    "planned_weight", "status", "entry_price", "exit_price", "gross_return",
    "net_return", "return_contribution", "turnover", "buy_notional", "sell_notional",
    "buy_cost", "sell_cost",
]


def _normalize_keys(table, name):
    table = table.copy()
    table["Date"] = pd.to_datetime(table["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    table["Ticker"] = table["Ticker"].astype("string").str.strip().str.upper().str.replace(".", "-", regex=False)
    invalid = table["Date"].isna() | table["Ticker"].isna() | table["Ticker"].eq("")
    if invalid.any():
        raise ValueError(f"{name} contains invalid Date/Ticker keys.")
    if table.duplicated(["Date", "Ticker"]).any():
        raise ValueError(f"{name} contains duplicate Date/Ticker keys.")
    return table


def _checked_inputs(frame, predictions, prediction_column, start_date):
    required = {"Date", "Ticker", "SignalEligible", "Open", "Close", "target", "LabelEndDate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Quote panel is missing columns: {sorted(missing)}")
    prediction_required = {"Date", "Ticker", prediction_column}
    missing = prediction_required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    execution_columns = [column for column in ["ExecutionOpen", "ExecutionClose"] if column in frame]
    panel = _normalize_keys(frame[list(required) + execution_columns], "Quote panel")
    if panel.empty:
        raise ValueError("Quote panel is empty.")
    # A string such as 'False' must not silently become True.
    if panel["SignalEligible"].isna().any() or not panel["SignalEligible"].isin([True, False]).all():
        raise ValueError("SignalEligible must contain nonmissing boolean values.")
    panel["SignalEligible"] = panel["SignalEligible"].astype(bool)
    boundary = pd.to_datetime(start_date, errors="coerce", utc=True)
    if pd.isna(boundary):
        raise ValueError("start_date must be a valid date.")
    boundary = boundary.tz_localize(None).normalize()
    calendar = pd.DatetimeIndex(panel["Date"].unique()).sort_values()
    test_calendar = calendar[calendar >= boundary]
    if test_calendar.empty:
        raise ValueError("No calendar sessions occur on or after start_date.")
    submitted = _normalize_keys(predictions[["Date", "Ticker", prediction_column]], "Predictions")
    submitted = submitted.rename(columns={prediction_column: "prediction"})
    submitted["prediction"] = pd.to_numeric(submitted["prediction"], errors="coerce")
    if not np.isfinite(submitted["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("Predictions must contain finite numeric values.")
    eligible = panel.loc[panel["Date"].ge(boundary) & panel["SignalEligible"]]
    expected_keys = pd.MultiIndex.from_frame(eligible[["Date", "Ticker"]])
    actual_keys = pd.MultiIndex.from_frame(submitted[["Date", "Ticker"]])
    missing_keys = expected_keys.difference(actual_keys)
    extra_keys = actual_keys.difference(expected_keys)
    if len(missing_keys) or len(extra_keys):
        raise ValueError(
            "Prediction coverage mismatch: "
            f"{len(missing_keys)} eligible signals missing; "
            f"{len(extra_keys)} unknown, ineligible, or out-of-period signals supplied."
        )
    joined = submitted.merge(
        eligible[["Date", "Ticker", "target", "LabelEndDate"]],
        on=["Date", "Ticker"], how="left", validate="one_to_one",
    ).sort_values(["Date", "Ticker"])
    joined["target"] = pd.to_numeric(joined["target"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    joined["LabelEndDate"] = pd.to_datetime(joined["LabelEndDate"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    # A retained label beyond the supplied snapshot is not observed evidence.
    joined.loc[
        joined["LabelEndDate"].isna()
        | joined["LabelEndDate"].le(joined["Date"])
        | joined["LabelEndDate"].gt(calendar[-1]), "target"
    ] = np.nan
    return panel, joined, calendar, test_calendar


def _rank_scores(joined):
    rows = []
    for date, signals in joined.groupby("Date", sort=True):
        labeled = signals.loc[signals["target"].notna()]
        defined = len(labeled) >= 2 and labeled["prediction"].nunique() >= 2 and labeled["target"].nunique() >= 2
        ic = labeled["prediction"].rank().corr(labeled["target"].rank()) if defined else np.nan
        rows.append({"Date": date, "rank_ic": ic, "prediction_rows": len(signals), "labeled_rows": len(labeled)})
    scores = pd.DataFrame(rows, columns=["Date", "rank_ic", "prediction_rows", "labeled_rows"])
    valid = scores["rank_ic"].dropna()
    return {
        "rank_ic_mean": float(valid.mean()) if len(valid) else np.nan,
        "rank_ic_std": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
        "rank_ic_scored_dates": int(len(valid)),
        "rank_ic_undefined_dates": int(len(scores) - len(valid)),
        "prediction_rows": int(len(joined)),
        "labeled_prediction_rows": int(joined["target"].notna().sum()),
        "unlabeled_prediction_rows": int(joined["target"].isna().sum()),
        "prediction_coverage": 1.0,
    }, scores


def score_predictions(frame, predictions, *, prediction_column="prediction", start_date="2024-01-01"):
    """Return daily rank-IC statistics, enforcing complete eligible coverage.

    This helper shares the evaluator's coverage and label-availability checks.
    Use a panel truncated to the validation endpoint when scoring validation.
    """
    _, joined, _, _ = _checked_inputs(frame, predictions, prediction_column, start_date)
    return _rank_scores(joined)


def _positive_price(value):
    return np.isfinite(value) and value > 0


def _portfolio_session(selected, quotes, signal_date, holding_date, weight, cost, portfolio):
    trades = []
    gross, net, turnover = 0.0, 0.0, 0.0
    executed, missing, unresolved = 0, 0, False
    for ticker, prediction in selected:
        entry, exit_price = quotes.get((holding_date, ticker), (np.nan, np.nan))
        trade = {
            "portfolio": portfolio, "SignalDate": signal_date, "HoldingDate": holding_date,
            "Ticker": ticker, "prediction": prediction, "planned_weight": weight,
            "entry_price": entry, "exit_price": exit_price,
            "gross_return": np.nan, "net_return": np.nan, "return_contribution": np.nan,
            "turnover": 0.0, "buy_notional": 0.0, "sell_notional": 0.0,
            "buy_cost": 0.0, "sell_cost": 0.0,
        }
        if not _positive_price(entry):
            trade.update(status="missing_entry", gross_return=0.0, net_return=0.0, return_contribution=0.0)
            missing += 1
        else:
            executed += 1
            # Each slot is a capital budget INCLUDING its buy commission.
            # Shares * entry + buy commission = planned weight of capital.
            buy_notional = weight / (1 + cost)
            trade.update(buy_notional=buy_notional, buy_cost=cost * buy_notional)
            if not _positive_price(exit_price):
                trade.update(status="unresolved_exit", turnover=np.nan, sell_notional=np.nan, sell_cost=np.nan)
                unresolved = True
            else:
                ratio = exit_price / entry
                sell_notional = buy_notional * ratio
                contribution = sell_notional * (1 - cost) - weight
                trade.update(
                    status="executed", gross_return=ratio - 1,
                    net_return=ratio * (1 - cost) / (1 + cost) - 1,
                    return_contribution=contribution,
                    turnover=buy_notional + sell_notional,
                    sell_notional=sell_notional, sell_cost=cost * sell_notional,
                )
                gross += weight * (ratio - 1)
                net += contribution
                turnover += buy_notional + sell_notional
        trades.append(trade)
    return {
        "gross_return": np.nan if unresolved else gross,
        "net_return": np.nan if unresolved else net,
        "turnover": np.nan if unresolved else turnover,
        "unresolved": unresolved, "selected_count": len(selected), "executed_count": executed,
        "missing_entry_count": missing, "cash_weight": 1.0 - executed * weight,
    }, trades


def _metrics(gross, net, turnover):
    count = len(net)
    if not count:
        return {key: np.nan for key in [
            "gross_cumulative_return", "net_cumulative_return", "annualized_return",
            "annualized_sharpe", "annualized_volatility", "max_drawdown", "mean_daily_turnover",
        ]}
    wealth = np.concatenate(([1.0], np.cumprod(1 + net)))
    std = float(np.std(net, ddof=1)) if count > 1 else np.nan
    return {
        "gross_cumulative_return": float(np.prod(1 + gross) - 1),
        "net_cumulative_return": float(wealth[-1] - 1),
        "annualized_return": float(wealth[-1] ** (252 / count) - 1),
        "annualized_sharpe": float(np.mean(net) / std * np.sqrt(252)) if np.isfinite(std) and std > 0 else np.nan,
        "annualized_volatility": float(std * np.sqrt(252)) if np.isfinite(std) else np.nan,
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "mean_daily_turnover": float(np.mean(turnover)),
    }


def evaluate_predictions(
    frame, predictions, *, prediction_column="prediction", top_k=10,
    cost_bps=10.0, start_date="2024-01-01",
):
    """Return (summary, daily, trades) for complete, chronological predictions.

    Full coverage includes eligible last-date signals even though their next
    session lies outside the snapshot; those signals are counted and excluded
    from trading. Rank selection never consults labels or tomorrow's quotes.

    Optional ExecutionOpen/ExecutionClose preserve individually valid raw
    prices when the cleaner masks an entire invalid daily quote. They take
    precedence over Open/Close. Open availability is assessed at entry
    independently of the end-of-session QuoteValid flag. A missing entry keeps its original slot in cash, without
    replacements. A valid entry with an unknown exit marks the session
    unresolved. Summary metrics for BOTH portfolios use their common resolved
    sessions, with omissions explicitly counted; daily retains all sessions.

    Planned weights are capital budgets including buy costs, so the portfolio
    cannot spend more than its capital. One-way turnover sums executed buy and
    sell notionals divided by starting capital. Cash has zero return. Net
    Sharpe uses 252 sessions/year and a zero cash rate.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be finite and between 0 and 10000.")
    cost = float(cost_bps) / 10000
    panel, joined, calendar, test_calendar = _checked_inputs(frame, predictions, prediction_column, start_date)
    summary, scores = _rank_scores(joined)
    score_lookup = dict(zip(scores["Date"], scores["rank_ic"]))
    by_date = dict(tuple(joined.groupby("Date", sort=True)))
    entry_column = "ExecutionOpen" if "ExecutionOpen" in panel else "Open"
    exit_column = "ExecutionClose" if "ExecutionClose" in panel else "Close"
    future_quotes = panel.loc[panel["Date"].ge(test_calendar[0]), ["Date", "Ticker", entry_column, exit_column]].copy()
    future_quotes.columns = ["Date", "Ticker", "Open", "Close"]
    for column in ["Open", "Close"]:
        future_quotes[column] = pd.to_numeric(future_quotes[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    quote_lookup = {
        (date, ticker): (float(open_price), float(close_price))
        for date, ticker, open_price, close_price in future_quotes.itertuples(index=False, name=None)
    }
    successor = dict(zip(calendar[:-1], calendar[1:]))
    daily_rows, trade_rows = [], []
    for signal_date in test_calendar[:-1]:
        holding_date = successor[signal_date]
        signals = by_date.get(signal_date, joined.iloc[:0])
        ranked = signals.sort_values(["prediction", "Ticker"], ascending=[False, True], kind="stable")
        selected = list(ranked.head(top_k)[["Ticker", "prediction"]].itertuples(index=False, name=None))
        strategy, trades = _portfolio_session(selected, quote_lookup, signal_date, holding_date, 1 / top_k, cost, "strategy")
        trade_rows.extend(trades)
        all_signals = list(signals[["Ticker", "prediction"]].itertuples(index=False, name=None))
        benchmark, trades = _portfolio_session(
            all_signals, quote_lookup, signal_date, holding_date,
            1 / len(all_signals) if all_signals else 0.0, cost, "benchmark",
        )
        trade_rows.extend(trades)
        daily_rows.append({
            "SignalDate": signal_date, "HoldingDate": holding_date,
            **strategy, "rank_ic": score_lookup.get(signal_date, np.nan),
            **{f"benchmark_{key}": value for key, value in benchmark.items()},
        })
    daily = pd.DataFrame(daily_rows, columns=_DAILY_COLUMNS)
    trades = pd.DataFrame(trade_rows, columns=_TRADE_COLUMNS)
    resolved = daily.loc[~(daily["unresolved"].astype(bool) | daily["benchmark_unresolved"].astype(bool))]
    summary.update(_metrics(
        resolved["gross_return"].to_numpy(dtype=float), resolved["net_return"].to_numpy(dtype=float),
        resolved["turnover"].to_numpy(dtype=float),
    ))
    summary.update({f"benchmark_{key}": value for key, value in _metrics(
        resolved["benchmark_gross_return"].to_numpy(dtype=float), resolved["benchmark_net_return"].to_numpy(dtype=float),
        resolved["benchmark_turnover"].to_numpy(dtype=float),
    ).items()})
    final_signals = joined.loc[joined["Date"].eq(calendar[-1])]
    summary.update({
        "signal_dates": int(joined["Date"].nunique()), "calendar_test_dates": int(len(test_calendar)),
        "trading_dates": int(len(daily)), "resolved_trading_dates": int((~daily["unresolved"].astype(bool)).sum()),
        "unresolved_trading_dates": int(daily["unresolved"].sum()),
        "benchmark_unresolved_trading_dates": int(daily["benchmark_unresolved"].sum()),
        "aggregate_resolved_dates": int(len(resolved)), "aggregate_excluded_unresolved_dates": int(len(daily) - len(resolved)),
        "fully_resolved": len(daily) == len(resolved),
        "aggregate_scope": "Common resolved sessions only; omitted sessions make these conditional, not full-period returns.",
        "excluded_final_signal_dates": int(not final_signals.empty),
        "excluded_final_prediction_rows": int(len(final_signals)),
        "start_date": test_calendar[0].strftime("%Y-%m-%d"), "end_date": test_calendar[-1].strftime("%Y-%m-%d"),
        "holding_start_date": daily["HoldingDate"].min().strftime("%Y-%m-%d") if len(daily) else None,
        "holding_end_date": daily["HoldingDate"].max().strftime("%Y-%m-%d") if len(daily) else None,
        "cost_bps": float(cost_bps), "top_k": int(top_k),
        "missing_entry_slots": int(daily["missing_entry_count"].sum()),
        "benchmark_missing_entry_slots": int(daily["benchmark_missing_entry_count"].sum()),
        "mean_cash_weight": float(daily["cash_weight"].mean()) if len(daily) else np.nan,
        "benchmark_mean_cash_weight": float(daily["benchmark_cash_weight"].mean()) if len(daily) else np.nan,
    })
    return summary, daily, trades
