"""Persistent, cash-funded long/short simulation with next-session OHLC4 fills.

This is a research approximation: a day's OHLC4 is known only after that day,
so it is a simulated fill, not an executable quote or guaranteed execution.
Prices must already share one corporate-action adjustment basis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtests.showcase import _normalize_keys


@dataclass
class _Position:
    side: str
    quantity: float = 0.0
    cost_basis: float = 0.0
    collateral: float = 0.0
    short_proceeds: float = 0.0


@dataclass
class _Book:
    name: str
    cash: float
    cost_rate: float
    borrow_rate: float = 0.0
    positions: dict = field(default_factory=dict)
    margin_debt: float = 0.0
    fees: float = 0.0
    borrow_fees: float = 0.0
    margin_shortfall_events: int = 0
    insolvency_observed: bool = False

    def credit(self, amount):
        """Settle a cash release/obligation; short-loss debt is never free cash."""
        if amount >= 0:
            repayment = min(amount, self.margin_debt)
            self.margin_debt -= repayment
            self.cash += amount - repayment
        else:
            self.cash += amount
            if self.cash < -1e-8:
                self.margin_debt += -self.cash
                self.cash = 0.0
                self.margin_shortfall_events += 1
            elif self.cash < 0:
                self.cash = 0.0


def _inputs(panel, predictions, prediction_column, start_date):
    required = {"Date", "Ticker", "SignalEligible", "ExecutionAverage", "MarkClose"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Rank-hold quote panel is missing columns: {sorted(missing)}")
    missing = {"Date", "Ticker", prediction_column} - set(predictions)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    quotes = _normalize_keys(panel[list(required)], "Rank-hold quote panel")
    if quotes.empty:
        raise ValueError("Rank-hold quote panel is empty.")
    if quotes["SignalEligible"].isna().any() or not quotes["SignalEligible"].isin([True, False]).all():
        raise ValueError("SignalEligible must contain nonmissing boolean values.")
    quotes["SignalEligible"] = quotes["SignalEligible"].astype(bool)
    for column in ["ExecutionAverage", "MarkClose"]:
        values = pd.to_numeric(quotes[column], errors="coerce").astype(float)
        quotes[column] = values.where(np.isfinite(values) & values.gt(0))
    boundary = pd.to_datetime(start_date, errors="coerce", utc=True)
    if pd.isna(boundary):
        raise ValueError("start_date must be a valid date.")
    boundary = boundary.tz_localize(None).normalize()
    calendar = pd.DatetimeIndex(quotes["Date"].unique()).sort_values()
    calendar = calendar[calendar >= boundary]
    if calendar.empty:
        raise ValueError("No calendar sessions occur on or after start_date.")
    submitted = _normalize_keys(predictions[["Date", "Ticker", prediction_column]], "Predictions")
    submitted = submitted.rename(columns={prediction_column: "prediction"})
    submitted["prediction"] = pd.to_numeric(submitted["prediction"], errors="coerce").astype(float)
    if not np.isfinite(submitted["prediction"].to_numpy()).all():
        raise ValueError("Predictions must contain finite numeric values.")
    eligible = quotes.loc[quotes["Date"].ge(boundary) & quotes["SignalEligible"], ["Date", "Ticker"]]
    expected = pd.MultiIndex.from_frame(eligible)
    actual = pd.MultiIndex.from_frame(submitted[["Date", "Ticker"]])
    absent, extra = expected.difference(actual), actual.difference(expected)
    if len(absent) or len(extra):
        raise ValueError(f"Prediction coverage mismatch: {len(absent)} eligible signals missing; {len(extra)} unknown, ineligible, or out-of-period signals supplied.")
    quotes = quotes.loc[quotes["Date"].ge(calendar[0])]
    prices = {
        (date, ticker): (average, mark)
        for date, ticker, average, mark in quotes[["Date", "Ticker", "ExecutionAverage", "MarkClose"]].itertuples(index=False, name=None)
    }
    by_date = dict(tuple(submitted.sort_values(["Date", "Ticker"]).groupby("Date", sort=True)))
    return calendar, submitted, prices, by_date


def _ranks(signals, entry_k):
    """One canonical ranking makes opposite tails disjoint even under ties.

    Descending score uses ticker ascending for ties. Short ascending rank is
    the reverse of that total ordering, so tied tickers reverse on the short
    side. If the universe is smaller than 2*k, each side uses floor(n/2).
    """
    ordered = signals.sort_values(["prediction", "Ticker"], ascending=[False, True], kind="stable")
    tickers = ordered["Ticker"].tolist()
    scores = dict(zip(ordered["Ticker"], ordered["prediction"]))
    long_ranks = {ticker: i + 1 for i, ticker in enumerate(tickers)}
    short_ranks = {ticker: i + 1 for i, ticker in enumerate(reversed(tickers))}
    effective_k = min(entry_k, len(tickers) // 2)
    long_entries = tickers[:effective_k]
    short_entries = list(reversed(tickers[-effective_k:])) if effective_k else []
    return long_ranks, short_ranks, long_entries, short_entries, scores


def _plans(book, signal_date, execution_date, rank_data, exit_k, long_fraction, allow_additions, exit_unranked):
    long_ranks, short_ranks, longs, shorts, scores = rank_data
    orders, exiting = [], set()
    for ticker, position in sorted(book.positions.items()):
        ranks = long_ranks if position.side == "long" else short_ranks
        rank = ranks.get(ticker)
        if (rank is None and exit_unranked) or (rank is not None and rank > exit_k):
            exiting.add(ticker)
            orders.append({
                "SignalDate": signal_date, "ExecutionDate": execution_date, "Ticker": ticker,
                "side": position.side, "action": "exit", "quantity": position.quantity,
                "prediction": scores.get(ticker, np.nan), "rank": rank,
                "planned_budget": 0.0,
            })
    # No proceeds from tomorrow's planned exits fund tomorrow's entry budget.
    deployable = book.cash if book.margin_shortfall_events == 0 and not book.insolvency_observed else 0.0
    for side, candidates, fraction, ranks in [
        ("long", longs, long_fraction, long_ranks),
        ("short", shorts, 1 - long_fraction, short_ranks),
    ]:
        if not candidates or fraction == 0:
            continue
        budget = deployable * fraction / len(candidates)
        if budget <= 1e-8:
            continue
        for ticker in candidates:
            held = book.positions.get(ticker)
            if held and ticker not in exiting:
                if held.side != side or not allow_additions:
                    # Current ranks/holdings are known now; its slot stays cash.
                    continue
            orders.append({
                "SignalDate": signal_date, "ExecutionDate": execution_date, "Ticker": ticker,
                "side": side, "action": "entry", "quantity": np.nan,
                "prediction": scores[ticker], "rank": ranks[ticker],
                "planned_budget": budget,
            })
    return orders


def _trade_row(book, order, status, *, price=np.nan, quantity=0.0, notional=0.0, cost=0.0, cash_flow=0.0, funded_budget=0.0):
    buy = (order["side"] == "long" and order["action"] == "entry") or (order["side"] == "short" and order["action"] == "exit")
    return {
        "book": book.name, **order, "status": status, "fill_price": price,
        "quantity": quantity, "signed_quantity": quantity if buy else -quantity,
        "notional": notional, "cost": cost, "cash_flow": cash_flow,
        "funded_budget": funded_budget, "planned_quantity": order["quantity"],
    }


def _execute(book, orders, prices, date):
    rows, notional_total, costs = [], 0.0, 0.0
    total_exit_cash_flow = 0.0
    # Exit obligations settle first, but the entry plans are already fixed.
    for order in [item for item in orders if item["action"] == "exit"]:
        position = book.positions.get(order["Ticker"])
        if position is None or position.side != order["side"]:
            raise RuntimeError("Exit plan no longer matches inventory.")
        price = prices.get((date, order["Ticker"]), (np.nan, np.nan))[0]
        if not np.isfinite(price):
            rows.append(_trade_row(book, order, "missing_exit_fill"))
            continue
        quantity, notional = position.quantity, position.quantity * price
        fee = notional * book.cost_rate
        release = notional - fee if position.side == "long" else position.collateral + position.short_proceeds - notional - fee
        total_exit_cash_flow += release
        book.fees += fee
        notional_total += notional
        costs += fee
        del book.positions[order["Ticker"]]
        rows.append(_trade_row(book, order, "filled", price=price, quantity=quantity, notional=notional, cost=fee, cash_flow=release))
    book.credit(total_exit_cash_flow)
    entries = [item for item in orders if item["action"] == "entry"]
    total_planned = sum(order["planned_budget"] for order in entries)
    # Caps only enforce actual solvency after an unexpected cover obligation.
    # No candidate replacement or allocation to missed quotes occurs.
    fraction = min(1.0, book.cash / total_planned) if total_planned > 0 and book.margin_shortfall_events == 0 and not book.insolvency_observed else 0.0
    for order in entries:
        price = prices.get((date, order["Ticker"]), (np.nan, np.nan))[0]
        held = book.positions.get(order["Ticker"])
        if not np.isfinite(price):
            rows.append(_trade_row(book, order, "missing_entry_fill"))
            continue
        if held and held.side != order["side"]:
            rows.append(_trade_row(book, order, "blocked_opposite_position", price=price))
            continue
        budget = order["planned_budget"] * fraction
        if budget <= 1e-8:
            rows.append(_trade_row(book, order, "unfunded", price=price))
            continue
        notional = budget / (1 + book.cost_rate)
        fee, quantity = notional * book.cost_rate, notional / price
        if not np.isfinite(notional) or not np.isfinite(quantity):
            raise ValueError(f"Nonfinite executed position for {order['Ticker']} on {date.date()}.")
        book.cash -= notional + fee
        if book.cash < -1e-6:
            raise RuntimeError("Entry spends more than available cash.")
        book.cash = max(0.0, book.cash)
        position = held or _Position(order["side"])
        position.quantity += quantity
        position.cost_basis += notional
        if position.side == "short":
            position.collateral += notional
            position.short_proceeds += notional
        book.positions[order["Ticker"]] = position
        book.fees += fee
        notional_total += notional
        costs += fee
        status = "partially_funded" if fraction < 1 - 1e-12 else "filled"
        rows.append(_trade_row(book, order, status, price=price, quantity=quantity, notional=notional, cost=fee, cash_flow=-budget, funded_budget=budget))
    return rows, notional_total, costs


def _mark(book, prices, date):
    long_value, short_liability, collateral, proceeds = 0.0, 0.0, 0.0, 0.0
    positions = []
    for ticker, position in sorted(book.positions.items()):
        mark = prices.get((date, ticker), (np.nan, np.nan))[1]
        if not np.isfinite(mark):
            raise ValueError(f"Unresolved held mark: {book.name} {position.side} {ticker} on {date.date()}. No stale-price or zero-return substitution is allowed.")
        market_value = position.quantity * mark
        if not np.isfinite(market_value):
            raise ValueError(f"Nonfinite held market value for {ticker} on {date.date()}.")
        if position.side == "long":
            long_value += market_value
            unrealized = market_value - position.cost_basis
        else:
            short_liability += market_value
            collateral += position.collateral
            proceeds += position.short_proceeds
            unrealized = position.short_proceeds - market_value
        positions.append({
            "Date": date, "book": book.name, "Ticker": ticker, "side": position.side,
            "quantity": position.quantity, "signed_quantity": position.quantity if position.side == "long" else -position.quantity,
            "mark_price": mark, "market_value": market_value, "cost_basis": position.cost_basis,
            "collateral": position.collateral, "short_proceeds": position.short_proceeds,
            "unrealized_pnl_before_fees": unrealized,
        })
    borrow = short_liability * book.borrow_rate / 252
    if borrow:
        book.credit(-borrow)
        book.borrow_fees += borrow
    nav = book.cash - book.margin_debt + long_value + collateral + proceeds - short_liability
    if nav <= 0:
        book.insolvency_observed = True
    status = ("insolvent" if nav <= 0 else "post_insolvency" if book.insolvency_observed
              else "margin_shortfall" if book.margin_debt > 1e-8
              else "funding_shortfall_observed" if book.margin_shortfall_events else "resolved")
    values = {
        "nav": nav, "free_cash": book.cash, "margin_debt": book.margin_debt,
        "long_value": long_value, "short_liability": short_liability,
        "short_collateral": collateral, "short_proceeds": proceeds,
        "long_count": sum(p.side == "long" for p in book.positions.values()),
        "short_count": sum(p.side == "short" for p in book.positions.values()),
        "gross_exposure": (long_value + short_liability) / nav if nav > 0 else np.nan,
        "net_exposure": (long_value - short_liability) / nav if nav > 0 else np.nan,
        "borrow_cost": borrow, "status": status,
        "performance_valid": book.margin_shortfall_events == 0 and not book.insolvency_observed,
    }
    return values, positions


def _book_metrics(daily, name, initial_capital):
    nav = daily[f"{name}_nav"].to_numpy(dtype=float)
    returns = daily[f"{name}_return"].iloc[1:].to_numpy(dtype=float)
    defined = bool(np.isfinite(returns).all())
    std = float(np.std(returns, ddof=1)) if defined and len(returns) > 1 else np.nan
    count = len(returns)
    return {
        "cumulative_return": float(nav[-1] / initial_capital - 1),
        "annualized_return": float((nav[-1] / initial_capital) ** (252 / count) - 1) if count and nav[-1] > 0 and defined else np.nan,
        "annualized_sharpe": float(np.mean(returns) / std * np.sqrt(252)) if np.isfinite(std) and std > 0 else np.nan,
        "annualized_volatility": float(std * np.sqrt(252)) if np.isfinite(std) else np.nan,
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "mean_daily_turnover": float(daily[f"{name}_turnover"].iloc[1:].mean()) if count else np.nan,
        "final_nav": float(nav[-1]), "return_path_defined": defined,
    }


def evaluate_rank_hold(
    panel, predictions, *, prediction_column="prediction", initial_capital=1_000_000,
    entry_k=20, exit_k=100, cost_bps=10.0, start_date="2024-01-01",
    long_fraction=0.5, allow_additions=True, exit_unranked=True, annual_borrow_bps=0.0,
    price_units="supplied execution and mark prices on the same corporate-action adjustment basis",
):
    """Return summary, daily NAV, trade/order audit, and daily positions.

    Rank at the completed session close. Enter the top/bottom k at the next
    observed session's supplied ExecutionAverage; retain each side until its
    own rank exceeds exit_k (or it is unranked if exit_unranked=True).
    Mark at that session's supplied MarkClose. No forced final liquidation.

    Only free cash already known on SignalDate funds entry plans, split by
    long_fraction and equally among that day's selected tail. Existing
    same-side positions may receive additions. Capital released by an exit
    on d can be allocated at close d for execution on d+1, never sooner.
    Gross and net books use separate cash, shares, collateral and returns.

    Each short entry consumes its own collateral plus commission. Short-sale
    proceeds are segregated and never reinvested. A cover loss exceeding
    collateral and free cash is an explicit margin debt: it lowers NAV,
    blocks all subsequent new risk, and is repaid by later cash releases. Any
    shortfall or insolvency invalidates performance metrics for that book; a
    later accounting recovery does not restore validity. No margin call or
    forced liquidation is modeled. Missing held marks fail transparently.
    All selected stocks are assumed shortable; default borrow cost is zero.
    """
    for name, value in [("entry_k", entry_k), ("exit_k", exit_k)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if exit_k < entry_k:
        raise ValueError("exit_k must be at least entry_k.")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be finite and between 0 and 10000.")
    if not np.isfinite(long_fraction) or not 0 <= long_fraction <= 1:
        raise ValueError("long_fraction must be between zero and one.")
    if not np.isfinite(annual_borrow_bps) or annual_borrow_bps < 0:
        raise ValueError("annual_borrow_bps must be finite and nonnegative.")
    calendar, submitted, prices, by_date = _inputs(panel, predictions, prediction_column, start_date)
    books = {
        "gross": _Book("gross", float(initial_capital), 0.0),
        "net": _Book("net", float(initial_capital), float(cost_bps) / 10000, float(annual_borrow_bps) / 10000),
    }
    queued = {name: [] for name in books}
    prior_nav = {name: initial_capital for name in books}
    daily_rows, trade_rows, position_rows = [], [], []
    for index, date in enumerate(calendar):
        row = {"Date": date}
        signals = by_date.get(date, submitted.iloc[:0])
        rank_data = _ranks(signals, entry_k)
        row["eligible_count"] = len(signals)
        row["effective_entry_k"] = len(rank_data[2])
        following = calendar[index + 1] if index + 1 < len(calendar) else pd.NaT
        for name, book in books.items():
            trades, turnover_notional, fees = _execute(book, queued[name], prices, date)
            trade_rows.extend(trades)
            marked, positions = _mark(book, prices, date)
            position_rows.extend(positions)
            row.update({f"{name}_{key}": value for key, value in marked.items()})
            row[f"{name}_return"] = marked["nav"] / prior_nav[name] - 1 if index > 0 and prior_nav[name] > 0 else np.nan
            row[f"{name}_turnover"] = turnover_notional / prior_nav[name] if prior_nav[name] > 0 else np.nan
            row[f"{name}_transaction_cost"] = fees
            row[f"{name}_cumulative_transaction_cost"] = book.fees
            plans = _plans(book, date, following, rank_data, exit_k, long_fraction, bool(allow_additions), bool(exit_unranked))
            row[f"{name}_planned_entry_budget"] = sum(item["planned_budget"] for item in plans)
            row[f"{name}_planned_exit_count"] = sum(item["action"] == "exit" for item in plans)
            if pd.isna(following):
                trade_rows.extend(_trade_row(book, item, "pending_beyond_snapshot") for item in plans)
                queued[name] = []
            else:
                queued[name] = plans
            prior_nav[name] = marked["nav"]
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    trade_columns = [
        "book", "SignalDate", "ExecutionDate", "Ticker", "side", "action", "prediction",
        "rank", "planned_budget", "status", "fill_price", "quantity", "signed_quantity",
        "notional", "cost", "cash_flow", "funded_budget", "planned_quantity",
    ]
    position_columns = [
        "Date", "book", "Ticker", "side", "quantity", "signed_quantity", "mark_price",
        "market_value", "cost_basis", "collateral", "short_proceeds", "unrealized_pnl_before_fees",
    ]
    trades = pd.DataFrame(trade_rows, columns=trade_columns)
    positions = pd.DataFrame(position_rows, columns=position_columns)
    gross, net = _book_metrics(daily, "gross", initial_capital), _book_metrics(daily, "net", initial_capital)
    gross_valid = books["gross"].margin_shortfall_events == 0 and not books["gross"].insolvency_observed
    net_valid = books["net"].margin_shortfall_events == 0 and not books["net"].insolvency_observed
    for metrics, valid in [(gross, gross_valid), (net, net_valid)]:
        if not valid:
            for key in ["cumulative_return", "annualized_return", "annualized_sharpe", "annualized_volatility", "max_drawdown"]:
                metrics[key] = np.nan
    summary = {
        "initial_capital": float(initial_capital), "entry_k": int(entry_k), "exit_k": int(exit_k),
        "long_fraction": float(long_fraction), "short_fraction": float(1 - long_fraction),
        "allow_additions": bool(allow_additions), "exit_unranked": bool(exit_unranked),
        "cost_bps": float(cost_bps), "annual_borrow_bps": float(annual_borrow_bps),
        "start_date": calendar[0].strftime("%Y-%m-%d"), "end_date": calendar[-1].strftime("%Y-%m-%d"),
        "first_execution_date": calendar[1].strftime("%Y-%m-%d") if len(calendar) > 1 else None,
        "calendar_sessions": int(len(calendar)), "return_sessions": int(len(calendar) - 1),
        "strategy_name": "rank_hold_long_short",
        "trading_dates": int(len(calendar) - 1), "aggregate_resolved_dates": int(len(calendar) - 1),
        "aggregate_excluded_unresolved_dates": 0,
        "final_net_equity": net["final_nav"], "final_gross_equity": gross["final_nav"],
        "prediction_rows": int(len(submitted)), "prediction_coverage": 1.0,
        "gross_cumulative_return": gross["cumulative_return"], "net_cumulative_return": net["cumulative_return"],
        **{f"gross_{key}": value for key, value in gross.items() if key != "cumulative_return"},
        **{key: value for key, value in net.items() if key not in ["cumulative_return", "final_nav"]},
        "net_final_nav": net["final_nav"],
        "excluded_final_signal_dates": int(calendar[-1] in by_date),
        "pending_final_orders": int(trades["status"].eq("pending_beyond_snapshot").sum()),
        "missing_entry_orders": int(trades["status"].eq("missing_entry_fill").sum()),
        "missing_exit_orders": int(trades["status"].eq("missing_exit_fill").sum()),
        "gross_final_long_count": int(daily.iloc[-1]["gross_long_count"]),
        "gross_final_short_count": int(daily.iloc[-1]["gross_short_count"]),
        "net_final_long_count": int(daily.iloc[-1]["net_long_count"]),
        "net_final_short_count": int(daily.iloc[-1]["net_short_count"]),
        "gross_transaction_costs": float(books["gross"].fees), "net_transaction_costs": float(books["net"].fees),
        "net_borrow_costs": float(books["net"].borrow_fees),
        "gross_margin_shortfall_events": books["gross"].margin_shortfall_events,
        "net_margin_shortfall_events": books["net"].margin_shortfall_events,
        "gross_insolvency_observed": books["gross"].insolvency_observed,
        "net_insolvency_observed": books["net"].insolvency_observed,
        "quote_resolution_complete": True,
        "funding_valid": books["gross"].margin_shortfall_events == 0 and books["net"].margin_shortfall_events == 0,
        "gross_performance_valid": gross_valid, "net_performance_valid": net_valid,
        "performance_valid": gross_valid and net_valid, "fully_resolved": gross_valid and net_valid,
        "performance_invalidation_reason": None if gross_valid and net_valid else "A funding shortfall or insolvency required unmodeled external credit; accounting NAV remains auditable, but performance statistics are invalid.",
        "price_units": str(price_units),
        "allocation_policy": "Only cash known at signal close funds next-session entries; split by side, then equally among selected names. Exit releases can fund the following signal, not same-session buys.",
        "short_accounting": "Short collateral and sale proceeds are segregated; neither finances new entries. NAV subtracts marked short liabilities and any margin-shortfall debt.",
        "tie_policy": "Descending score then ticker ascending; short order reverses the same total ranking, ensuring disjoint tails.",
        "small_universe_policy": "Each entry tail contains min(entry_k, floor(eligible_count/2)) securities.",
        "terminal_policy": "Open positions marked at final close; no forced liquidation or invented terminal commission.",
        "limitations": [
            "OHLC4 is a simulated average known only after the execution day, not a guaranteed attainable trade price.",
            "All selected stocks are assumed shortable. Locate failures, recalls and real margin requirements are unmodeled.",
            "Borrow fees default to zero; fees, spread and market impact need stronger research assumptions.",
            "Rank-based exits can retain positions for long periods; no stop-loss, forced margin liquidation or liquidity limit is modeled.",
            "Any emergency external-credit shortfall or insolvency invalidates headline performance; later ledger recovery is not a valid strategy recovery.",
            "The supplied adjusted price basis must be consistent for fills, marks and historical inventory.",
        ],
    }
    return summary, daily, trades, positions
