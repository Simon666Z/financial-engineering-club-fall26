"""Render recorded showcase results as a self-contained local dashboard."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

INK, GREEN, AMBER, MUTED, PAPER = "#233a33", "#216c58", "#ac733c", "#78817c", "#fbfaf6"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any, decimals: int = 2, percent: bool = False) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        return f"{value * 100:,.{decimals}f}%" if percent else f"{value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def compact(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def date(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%d %b %Y") if value is not None else "—"
    except (TypeError, ValueError):
        return escape(value)


def dates_frame(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    if "Date" not in frame:
        for name in ["HoldingDate", "date", "holding_date", "SignalDate"]:
            if name in frame:
                frame["Date"] = frame[name]
                break
    if "Date" not in frame:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame["Date"] = frame.index
        else:
            return pd.DataFrame(columns=["Date"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.dropna(subset=["Date"]).sort_values("Date")


def finite_series(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in frame:
        return pd.DataFrame(columns=["Date", name])
    out = frame[["Date", name]].copy()
    out[name] = pd.to_numeric(out[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=[name])


def resolved_portfolio_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the evaluator's common resolved sessions for both portfolio paths."""
    keep = pd.Series(True, index=frame.index)
    for flag in ["unresolved", "benchmark_unresolved"]:
        if flag in frame:
            keep &= ~frame[flag].astype(str).str.lower().isin(["true", "1"])
    for name in ["gross_return", "net_return", "benchmark_gross_return", "benchmark_net_return"]:
        if name in frame:
            keep &= np.isfinite(pd.to_numeric(frame[name], errors="coerce"))
    return frame.loc[keep].copy()


def format_dates(fig: Any, ax: Any) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate(rotation=0)


def style_axes(ax: Any) -> None:
    ax.set_facecolor(PAPER)
    ax.grid(axis="y", color="#e4e7df", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9ded5")
    ax.set_ylabel(ax.get_ylabel(), color=MUTED, fontsize=9)


def save_chart(fig: Any, directory: Path, name: str, label: str) -> str:
    directory = directory / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    svg_path = directory / f"{name}.svg"
    fig.savefig(svg_path, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", facecolor=PAPER, bbox_inches="tight", dpi=160)
    plt.close(fig)
    svg = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n"
    svg_path.write_text(svg, encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    return svg.replace("<svg ", f'<svg role="img" aria-label="{escape(label)}" ', 1)


def empty_chart(message: str) -> str:
    return f'<div class="chart-empty">{escape(message)}</div>'


def equity_chart(frame: pd.DataFrame, directory: Path, kind: str) -> str:
    fig, ax = plt.subplots(figsize=(9.7, 3.8))
    plotted = 0
    for name, label, color in [
        (f"{kind}_return", "Showcase portfolio", GREEN),
        (f"benchmark_{kind}_return", "Equal-weight benchmark", AMBER),
    ]:
        values = finite_series(frame, name)
        if not values.empty:
            ax.plot(values["Date"], (1 + values[name]).cumprod(), color=color, linewidth=1.7, label=label)
            plotted += 1
    if not plotted:
        plt.close(fig)
        return empty_chart("No resolved portfolio returns were recorded.")
    ax.axhline(1, color="#bcc9be", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Growth of 1 unit")
    style_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    format_dates(fig, ax)
    return save_chart(fig, directory, f"equity_{kind}", f"Recorded {kind} equity versus equal-weight benchmark")


def drawdown_chart(frame: pd.DataFrame, directory: Path) -> str:
    values = finite_series(frame, "net_return")
    if values.empty:
        return empty_chart("No resolved net returns were recorded.")
    wealth = (1 + values["net_return"]).cumprod().to_numpy()
    high_water = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdown = wealth / high_water - 1
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.fill_between(values["Date"], drawdown * 100, 0, color=GREEN, alpha=0.16)
    ax.plot(values["Date"], drawdown * 100, color=GREEN, linewidth=1.2)
    ax.set_ylabel("Drawdown (%)")
    style_axes(ax)
    format_dates(fig, ax)
    return save_chart(fig, directory, "drawdown_net", "Drawdown of recorded net portfolio returns")


def ic_chart(frame: pd.DataFrame, directory: Path) -> str:
    values = finite_series(frame, "rank_ic")
    if values.empty:
        return empty_chart("Daily rank correlation was not recorded.")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(values["Date"], values["rank_ic"], color="#b7c9c0", linewidth=0.65, alpha=0.85, label="Daily")
    ax.plot(values["Date"], values["rank_ic"].rolling(20, min_periods=5).mean(), color=GREEN, linewidth=1.6, label="20-observation mean")
    ax.axhline(0, color=AMBER, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Spearman rank correlation")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, labelcolor=INK)
    format_dates(fig, ax)
    return save_chart(fig, directory, "rank_ic", "Daily rank correlation and trailing twenty-observation average")


def importance_chart(importance: pd.DataFrame, directory: Path) -> str:
    if not {"feature", "importance"}.issubset(importance.columns):
        return empty_chart("Feature importance was not exported for this experiment.")
    values = importance[["feature", "importance"]].copy()
    values["importance"] = pd.to_numeric(values["importance"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna().sort_values("importance").tail(12)
    if values.empty:
        return empty_chart("No finite feature-importance measurements were recorded.")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.barh(values["feature"].astype(str), values["importance"], color=GREEN, height=0.58)
    ax.set_xlabel("Exported model importance", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e4e7df", linewidth=0.7)
    return save_chart(fig, directory, "feature_importance", "Recorded feature importance")


def table(frame: pd.DataFrame, columns: list[tuple[str, str, str]], highlight: bool = False) -> str:
    columns = [column for column in columns if column[0] in frame]
    if frame.empty or not columns:
        return '<p class="muted">No measurements were recorded.</p>'
    head = "".join(f'<th scope="col">{label}</th>' for _, label, _ in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, kind in columns:
            value = row[key]
            if kind == "text":
                rendered = escape(value)
            elif kind == "integer":
                rendered = compact(value)
            else:
                rendered = number(value, 4 if kind == "precise" else 3 if key in {"rank_ic", "score", "disagreement"} else 2, kind == "percent")
            cells.append(f"<td>{rendered}</td>")
        css = ' class="highlight"' if highlight and "ensemble" in str(row.get("model", "")).lower() else ""
        rows.append(f"<tr{css}>{''.join(cells)}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def selected_metrics(summary: dict, leaderboard: pd.DataFrame) -> dict:
    if not leaderboard.empty:
        names = leaderboard.get("model", pd.Series("", index=leaderboard.index)).astype(str)
        candidates = leaderboard.loc[names.str.contains("ensemble", case=False, regex=False)]
        if not candidates.empty:
            return candidates.iloc[-1].to_dict()
    evaluation = summary.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return {}
    return {
        "rank_ic": evaluation.get("rank_ic_mean"),
        "cumulative_gross_return": evaluation.get("gross_cumulative_return"),
        "cumulative_net_return": evaluation.get("net_cumulative_return"),
        "sharpe_net": evaluation.get("annualized_sharpe"),
        "max_drawdown_net": evaluation.get("max_drawdown"),
    }


def verdict_panel(metrics: dict, leaderboard: pd.DataFrame) -> str:
    """Describe the measured result, including losses and stronger baselines."""
    def finite(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    net = finite(metrics.get("cumulative_net_return"))
    gross = finite(metrics.get("cumulative_gross_return"))
    ic = finite(metrics.get("rank_ic"))
    if net is None:
        title = "The portfolio verdict is not yet measured."
    elif net < 0:
        title = "The selected forecast loses after costs."
    else:
        title = "The selected forecast has a positive recorded net return."
    observations = []
    if ic is not None:
        phrase = "Forecast alignment is small" if abs(ic) < 0.02 else "Recorded forecast alignment"
        observations.append(f"<p><b>{phrase}.</b> Mean daily rank IC is {number(ic, 3)}; this is a forecast-ordering measurement, not a trading return.</p>")
    if net is not None and gross is not None:
        lead = "Commissions overwhelm the gross result" if gross > 0 and net < 0 else "Costs change the portfolio result"
        observations.append(f"<p><b>{lead}.</b> Gross return is {number(gross, 1, True)} and net return is {number(net, 1, True)} under the recorded daily trading policy.</p>")
    if net is not None and {"model", "cumulative_net_return"}.issubset(leaderboard.columns):
        simple = leaderboard.loc[leaderboard["model"].astype(str).isin(["reversal", "momentum", "elastic_net"])].copy()
        simple["cumulative_net_return"] = pd.to_numeric(simple["cumulative_net_return"], errors="coerce")
        simple = simple.dropna(subset=["cumulative_net_return"]).sort_values("cumulative_net_return", ascending=False)
        if not simple.empty and float(simple.iloc[0]["cumulative_net_return"]) > net:
            best = simple.iloc[0]
            observations.append(f'<p><b>A simpler baseline does better.</b> {escape(best["model"])} records {number(best["cumulative_net_return"], 1, True)} net return versus {number(net, 1, True)} for the selected forecast.</p>')
    return f'<section class="verdict" aria-label="First frozen experiment verdict"><p class="eyebrow">First frozen demo experiment</p><h2>{title}</h2><div class="verdict-grid">{"".join(observations)}</div><p class="chart-note">Selection used 2023 validation data. This result is preserved without selecting a different winner on the 2024–2025 test.</p></section>'



def feature_input_counts(summary: dict) -> dict:
    """Count the recorded raw and percentile inputs without fixing a feature set."""
    training = summary.get("training", {})
    dataset = summary.get("dataset", {})
    schema = training.get("feature_schema", training.get("schema", {}))
    if not isinstance(schema, dict):
        schema = {}

    def names(value):
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        if isinstance(value, dict):
            for key in ["names", "features", "feature_names", "columns"]:
                if isinstance(value.get(key), (list, tuple)):
                    return [str(item) for item in value[key]]
        return []

    inputs = next((items for items in [
        names(training.get("feature_names")),
        names(schema.get("model_input_features")),
        names(training.get("input_features")),
        names(schema.get("input_features")),
        names(schema.get("model_input_features")),
        names(dataset.get("input_features")),
    ] if items), [])
    if inputs:
        percentiles = [name for name in inputs if name.endswith("__csrank")]
        raw = [name for name in inputs if not name.endswith("__csrank")]
        return {"raw": len(raw), "percentiles": len(percentiles), "total": len(inputs)}
    raw = next((items for items in [
        names(training.get("input_features")),
        names(training.get("original_features")),
        names(training.get("raw_features")),
        names(schema.get("raw_features")),
        names(dataset.get("features")),
    ] if items), [])
    percentiles = names(schema.get("percentile_features"))
    if raw:
        return {"raw": len(raw), "percentiles": len(percentiles) if percentiles else None,
                "total": len(raw) + len(percentiles) if percentiles else None}
    return {"raw": None, "percentiles": None, "total": None}


def costs_are_zero(evaluation: dict) -> bool:
    """Require recorded zero commissions and zero borrowing costs."""
    try:
        commission = float(evaluation.get("cost_bps", np.nan))
        borrow = float(evaluation.get("annual_borrow_bps", evaluation.get("net_borrow_costs", np.nan)))
        return commission == 0 and borrow == 0
    except (TypeError, ValueError):
        return False


def strategy_equity_chart(
    frame: pd.DataFrame, directory: Path, *, initial_capital: float | None = None, cost_free: bool = False,
) -> str:
    """Show actual strategy NAV when available, with an archived-return fallback."""
    fig, ax = plt.subplots(figsize=(9.7, 3.0))
    use_nav = {"gross_nav", "net_nav"}.issubset(frame.columns)
    if use_nav:
        # Both scenarios share the same visible valuation dates.
        frame = frame.copy()
        for column in ["gross_nav", "net_nav"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        keep = np.isfinite(frame["gross_nav"]) & np.isfinite(frame["net_nav"])
        for flag in ["unresolved", "benchmark_unresolved"]:
            if flag in frame:
                keep &= ~frame[flag].astype(str).str.lower().isin(["true", "1"])
        frame = frame.loc[keep]
    plotted = 0
    columns = [
        ("gross_nav" if use_nav else "gross_return", "Before costs", GREEN),
        ("net_nav" if use_nav else "net_return", "After costs", AMBER),
    ]
    if cost_free:
        columns = [("net_nav" if use_nav else "net_return", "Portfolio NAV", GREEN)]
    for name, label, color in columns:
        values = finite_series(frame, name)
        if not values.empty:
            equity = values[name] / 1_000_000 if use_nav else (1 + values[name]).cumprod()
            ax.plot(values["Date"], equity, color=color, linewidth=1.9, label=label)
            plotted += 1
    if not plotted:
        plt.close(fig)
        return empty_chart("No resolved strategy valuations were recorded.")
    baseline = float(initial_capital) / 1_000_000 if use_nav and initial_capital is not None else 1.0
    ax.axhline(baseline, color="#bcc9be", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Portfolio NAV ($m)" if use_nav else "Growth of 1 unit")
    style_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, fontsize=10, labelcolor=INK)
    format_dates(fig, ax)
    return save_chart(fig, directory, "strategy_equity",
                      "Selected strategy NAV with costs excluded" if cost_free else
                      "Selected strategy NAV before costs and after costs" if use_nav
                      else "Selected strategy equity before costs and after costs")


def render_showcase(
    report_dir: Path, summary: dict, leaderboard: pd.DataFrame, daily: pd.DataFrame,
    top_predictions: pd.DataFrame, feature_importance: pd.DataFrame,
) -> Path:
    """Write a simple model scheme and strategy-results dashboard."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = dates_frame(daily)
    portfolio_frame = resolved_portfolio_frame(frame)

    # Preserve the notebook's existing standalone chart exports.
    equity_chart(portfolio_frame, report_dir, "net")
    equity_chart(portfolio_frame, report_dir, "gross")
    drawdown_chart(portfolio_frame, report_dir)
    ic_chart(frame, report_dir)
    importance_chart(feature_importance, report_dir)
    evaluation = summary.get("evaluation", {})
    cost_free = costs_are_zero(evaluation)
    nav_frame = frame if {"gross_nav", "net_nav"}.issubset(frame.columns) else portfolio_frame
    strategy_equity = strategy_equity_chart(nav_frame, report_dir, initial_capital=evaluation.get("initial_capital"), cost_free=cost_free)
    training = summary.get("training", {})
    selected = selected_metrics(summary, leaderboard)
    weights = training.get("selected_weights", {}) if isinstance(training, dict) else {}
    names = {
        "elastic_net": "Elastic Net",
        "xgb_regression": "XGBoost regression",
        "xgb_ranker": "XGBoost Ranker",
        "catboost": "CatBoost",
    }
    active = [(name, weight) for name, weight in weights.items() if float(weight) > 0] if isinstance(weights, dict) else []
    if active:
        selected_name = "<br>".join(escape(names.get(name, name)) for name, _ in active)
        selected_weight = " · ".join(number(weight, 0, True) for _, weight in active)
    else:
        selected_name, selected_weight = "Selected forecast", "See saved model"
    selected_label = "Selected model" if len(active) == 1 else "Selected blend"
    def period_years(split, fallback):
        detail = summary.get("splits", {}).get(split, {})
        try:
            first, last = pd.Timestamp(detail["start"]).year, pd.Timestamp(detail["end"]).year
            return str(first) if first == last else f"{first}–{last}"
        except (KeyError, TypeError, ValueError):
            return fallback
    train_period = period_years("train", "through 2022")
    validation_period = period_years("validation", "2023")
    test_period = period_years("test", "2024–2025")
    feature_counts = feature_input_counts(summary)
    if feature_counts["raw"] is not None:
        input_label = f'{feature_counts["raw"]} raw features'
        if feature_counts["percentiles"] is not None:
            input_label += f'<br>+ {feature_counts["percentiles"]} daily percentiles'
    else:
        input_label = "Recorded feature schema"
    input_caption = "Research-backed price / volume signals."
    if feature_counts["total"] is not None:
        input_caption += f' {feature_counts["total"]} model inputs.'
    is_hold_strategy = evaluation.get("strategy_name") == "rank_hold_long_short" or {"gross_nav", "net_nav"}.issubset(frame.columns)
    top_k = compact(evaluation.get("top_k"))
    cost = number(evaluation.get("cost_bps"), 0)
    sessions = compact(evaluation.get("aggregate_resolved_dates", len(portfolio_frame)))
    cards = []
    metric_specs = [
        (evaluation.get("gross_cumulative_return", selected.get("cumulative_gross_return")), "Gross return", True),
        (evaluation.get("net_cumulative_return", selected.get("cumulative_net_return")), "Net return", True),
        (evaluation.get("annualized_sharpe", selected.get("sharpe_net")), "Net Sharpe", False),
        (evaluation.get("max_drawdown", selected.get("max_drawdown_net")), "Max drawdown", True),
    ]
    if cost_free:
        metric_specs = [
            (evaluation.get("net_cumulative_return", selected.get("cumulative_net_return")), "Return", True),
            (evaluation.get("annualized_sharpe", selected.get("sharpe_net")), "Sharpe", False),
            (evaluation.get("max_drawdown", selected.get("max_drawdown_net")), "Max drawdown", True),
        ]
    for value, label, percent in metric_specs:
        cards.append(f'<div class="metric"><span>{label}</span><strong>{number(value, 1 if percent else 2, percent)}</strong></div>')
    metrics_class = "metrics three" if cost_free else "metrics"
    gross_value = evaluation.get("gross_cumulative_return", selected.get("cumulative_gross_return"))
    net_value = evaluation.get("net_cumulative_return", selected.get("cumulative_net_return"))
    try:
        if float(gross_value) > 0 and float(net_value) < 0:
            outcome = "Trading costs overwhelm the gross gain: this strategy loses after costs."
        elif float(net_value) < 0:
            outcome = "This strategy loses after costs in the recorded backtest."
        elif float(net_value) >= 0:
            outcome = "The recorded strategy has a positive return after the stated costs."
        else:
            outcome = "Portfolio results are not yet available."
    except (TypeError, ValueError):
        outcome = "Portfolio results are not yet available."
    if cost_free:
        try:
            result = float(net_value)
            outcome = ("This recorded strategy gains with costs excluded." if result > 0 else
                       "This recorded strategy loses with costs excluded." if result < 0 else
                       "This recorded strategy breaks even with costs excluded." if result == 0 else
                       "Portfolio results are not yet available.")
        except (TypeError, ValueError):
            outcome = "Portfolio results are not yet available."
    if evaluation.get("performance_valid") is False:
        outcome = escape(evaluation.get("performance_invalidation_reason") or "Funding rules were breached; accounting NAV is auditable, but performance statistics are invalid.")
    excluded = int(evaluation.get("aggregate_excluded_unresolved_dates", 0) or 0)
    conditional = f" Results use common resolved sessions; {excluded:,} unresolved sessions are excluded." if excluded else ""
    if is_hold_strategy:
        strategy_title = "Top 20 long<br>Bottom 20 short"
        strategy_detail = "Hold while top / bottom 100.<br>Holdings can exceed 20 per side."
        strategy_caption = ("Prior close ranks → next-session adjusted OHLC4 fills. Deploy free cash 50% long / 50% short, "
                            "adding to top / bottom 20; hold while top / bottom 100. Exit proceeds become available at the fill-day close. "
                            f"Short proceeds and 100% entry collateral stay segregated. {cost} bps commission per side.")
        proxy_note = "OHLC4 is a fill proxy. Adjusted prices and units provide synthetic corporate-action accounting; actual cash dividends are not modeled."
        if "annual_borrow_bps" in evaluation:
            proxy_note += f" Borrow fee: {number(evaluation['annual_borrow_bps'], 0)} bps/year."
        capital = ('<div class="capital">'
                   f'<span>Initial <b>${number(evaluation.get("initial_capital"), 0)}</b></span>'
                   f'<span>Ending NAV before costs <b>${number(evaluation.get("final_gross_equity"), 0)}</b></span>'
                   f'<span>Ending NAV after costs <b>${number(evaluation.get("final_net_equity"), 0)}</b></span></div>')
        if cost_free:
            capital = ('<div class="capital">'
                       f'<span>Initial <b>${number(evaluation.get("initial_capital"), 0)}</b></span>'
                       f'<span>Ending NAV <b>${number(evaluation.get("final_net_equity"), 0)}</b></span></div>')
        footer = "User-defined hold strategy · Historical demonstration on the notebook universe."
    else:
        strategy_title = f"Rank stocks<br>→ Top {top_k}"
        strategy_detail = "Equal weight."
        strategy_caption = f"Signal at close → enter next open → exit that day's close. Cash overnight. {cost} bps commission per side; allocations fund buy costs."
        proxy_note, capital = "", ""
        footer = "First recorded experiment · Historical demonstration on the notebook universe."
    if cost_free:
        strategy_caption = strategy_caption.replace(f"{cost} bps commission per side.", "Costs excluded for this run.")
        strategy_caption = strategy_caption.replace(f"{cost} bps commission per side; allocations fund buy costs.", "Costs excluded for this run.")
        proxy_note = proxy_note.replace(" Borrow fee: 0 bps/year.", "")
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FE Club · Model &amp; Strategy</title>
<style>
:root{{--bg:#f1f2eb;--paper:#fbfaf6;--ink:#233a33;--muted:#69756e;--green:#216c58;--line:#dce1d5;--warm:#ac733c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;line-height:1.5}}
main{{max-width:1100px;margin:auto;padding:26px 28px 30px}}header{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:20px}}.brand{{font-size:13px;font-weight:650}}.brand span{{font-weight:400;color:var(--muted);margin-left:8px}}.period{{font-size:12px;color:var(--muted)}}section{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin-bottom:18px}}h1,h2{{margin:0;font-size:21px;font-weight:600;letter-spacing:-.02em}}.section-head{{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:18px}}.section-head p{{margin:0;color:var(--muted);font-size:12px}}.scheme{{display:grid;grid-template-columns:1fr 16px 1.45fr 16px .88fr 16px 1fr 16px 1.10fr;gap:8px;align-items:stretch}}.node{{border:1px solid var(--line);border-radius:7px;padding:15px 13px;display:flex;flex-direction:column;justify-content:center;min-height:156px}}.node-title{{font-size:12px;font-weight:600;color:var(--muted);margin:0 0 9px}}.node strong{{font-size:16px;font-weight:600;line-height:1.3}}.node p{{font-size:12px;line-height:1.5;color:var(--muted);margin:8px 0 0}}.arrow{{align-self:center;text-align:center;color:#87948b;font-size:22px}}.candidates{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}.candidate{{font-size:12px;line-height:1.4;border:1px solid var(--line);padding:7px 8px;border-radius:5px;background:#f4f5ef;display:flex;align-items:center;min-height:46px}}.selected{{background:#edf3e9;border-color:#b9cdbb}}.selected strong{{color:var(--green)}}.weight{{display:inline-block;align-self:flex-start;font-size:12px;color:var(--green);margin-top:9px;font-weight:600}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:7px;margin-bottom:16px}}.metrics.three{{grid-template-columns:repeat(3,1fr)}}.metric{{padding:14px 18px;border-right:1px solid var(--line)}}.metric:last-child{{border-right:0}}.metric span{{font-size:12px;color:var(--muted);display:block}}.metric strong{{font-weight:500;font-size:28px;letter-spacing:-.03em;display:block;margin-top:4px}}.chart svg{{display:block;width:100%;height:auto}}.chart-empty{{padding:80px 20px;text-align:center;color:var(--muted)}}.capital{{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:-3px 0 16px}}.capital b{{color:var(--ink);font-weight:600}}.proxy{{font-size:12px;color:var(--muted);line-height:1.5;margin:8px 0 0}}.strategy{{font-size:12px;line-height:1.6;color:var(--muted);margin:12px 0 0}}.outcome{{margin:11px 0 0;font-size:13px;color:#7e572e}}footer{{font-size:12px;color:var(--muted);margin:3px 0 0}}
@media(max-width:900px){{.scheme{{grid-template-columns:1fr 1.25fr 1fr;gap:12px}}.arrow{{display:none}}.node{{min-height:130px}}.section-head{{align-items:flex-start;flex-direction:column;gap:4px}}}}
@media(max-width:600px){{main{{padding:18px 14px}}section{{padding:18px 16px}}header{{align-items:flex-start;flex-direction:column;gap:5px}}.scheme{{grid-template-columns:1fr 1fr}}.node{{padding:12px;min-height:125px}}.candidates{{grid-template-columns:1fr}}.node.candidate-node{{grid-row:span 2}}.metrics{{grid-template-columns:1fr 1fr}}.metric:nth-child(2){{border-right:0}}.metric:nth-child(-n+2){{border-bottom:1px solid var(--line)}}.metric strong{{font-size:25px}}.metrics.three .metric:nth-child(2){{border-right:1px solid var(--line)}}.metrics.three .metric:nth-child(-n+2){{border-bottom:0}}.metrics.three .metric{{padding:12px}}}}
</style></head><body><main>
<header><div class="brand">FE CLUB <span>/ SimonResearch</span></div><div class="period">Train {train_period} · Validate {validation_period} · Test {test_period}</div></header>
<section aria-labelledby="scheme-title"><div class="section-head"><h1 id="scheme-title">Model scheme</h1><p>The four candidates are trained; validation selects the forecast.</p></div>
<div class="scheme" role="list" aria-label="Model and portfolio workflow">
<div class="node" role="listitem"><div class="node-title">Inputs</div><strong>{input_label}</strong><p>{input_caption}</p></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node candidate-node" role="listitem"><div class="node-title">Trained candidates</div><div class="candidates"><div class="candidate">Elastic Net</div><div class="candidate">XGBoost regression</div><div class="candidate">XGBoost Ranker</div><div class="candidate">CatBoost</div></div><p>Next-day return ranks<br><span>Ranker learns return deciles.</span></p></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node" role="listitem"><div class="node-title">Validation</div><strong>2023</strong><p>Select model settings and forecast weights.</p></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node selected" role="listitem"><div class="node-title">{selected_label}</div><strong>{selected_name}</strong><span class="weight">{selected_weight}</span></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node" role="listitem"><div class="node-title">Trading strategy</div><strong>{strategy_title}</strong><p>{strategy_detail}</p></div>
</div></section>
<section aria-labelledby="results-title"><div class="section-head"><h2 id="results-title">{test_period} backtest</h2><p>{sessions} trading sessions · Selected model strategy</p></div>{capital}<div class="{metrics_class}">{''.join(cards)}</div><div class="chart">{strategy_equity}</div><p class="strategy">{strategy_caption}{conditional}</p><p class="outcome">{outcome}</p><p class="proxy">{proxy_note}</p></section>
<footer>{footer}</footer>
</main></body></html>'''
    path = report_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
