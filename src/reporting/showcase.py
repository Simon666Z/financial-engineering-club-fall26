"""Render recorded showcase results as a self-contained local dashboard."""
from __future__ import annotations

import hashlib
import html
import io
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



def model_name(value: Any) -> str:
    """Use stable human names while allowing new learners and benchmarks."""
    name = str(value)
    return {
        "ensemble": "Blend",
        "blend": "Blend",
        "elastic_net": "Elastic Net",
        "xgb_regression": "XGBoost regression",
        "xgb_ranker": "XGBoost Ranker",
        "catboost": "CatBoost",
        "tabpfn": "TabPFN",
        "tabpfn_3_5": "TabPFN-3.5",
        "reversal": "Reversal baseline",
        "momentum": "Momentum baseline",
        "notebook_baseline": "Original-feature baseline",
        "benchmark": "Historical benchmark",
        "historical_benchmark": "Historical benchmark",
        "equal_weight_benchmark": "Equal-weight benchmark",
    }.get(name.lower(), name.replace("_", " ").strip())


def model_results_table(leaderboard: pd.DataFrame) -> str:
    """Keep every recorded model, including unresolved comparisons."""
    if leaderboard.empty or "model" not in leaderboard:
        return '<p class="muted">No model comparison was recorded.</p>'
    include_cost_return = "cumulative_return_after_costs" in leaderboard
    headers = ["Model", "Return", "Sharpe", "Max drawdown", "Ending NAV", "Status"]
    if include_cost_return:
        headers[1] = "Return before costs"
        headers.insert(2, "Return after costs")
    rows = []
    for _, row in leaderboard.iterrows():
        name = str(row["model"])
        status = str(row.get("status", "Recorded"))
        if status.lower() in {"nan", "", "none"}:
            status = "Recorded"
        invalid = status.lower() not in {"resolved", "valid", "complete", "recorded", "ok"}
        if "performance_valid" in row and str(row["performance_valid"]).lower() in {"false", "0"}:
            invalid = True
            if status.lower() in {"resolved", "valid", "complete", "recorded", "ok"}:
                status = "Invalid"
        values = [
            number(row.get("cumulative_net_return"), 1, True),
            number(row.get("sharpe_net"), 2),
            number(row.get("max_drawdown_net"), 1, True),
        ]
        ending = number(row.get("final_net_equity"), 0)
        values.append("$" + ending if ending != "—" else "—")
        if invalid:
            values = ["—"] * 4
        if include_cost_return:
            cost_status = str(row.get("cost_adjusted_status", "incomplete")).lower()
            cost_valid = cost_status in {"resolved", "valid", "complete", "recorded", "ok"}
            cost_return = number(row.get("cumulative_return_after_costs"), 1, True) if cost_valid else "—"
            if cost_return == "—":
                cost_failure = row.get("cost_adjusted_failure", "")
                reason = "" if pd.isna(cost_failure) else str(cost_failure)
                explanation = "Return after costs unavailable: " + cost_status.replace("_", " ")
                if reason:
                    explanation += ". " + reason
                cost_return = f'<span title="{escape(explanation)}" aria-label="{escape(explanation)}">—</span>'
            values.insert(1, cost_return)
        failure = row.get("failure", "")
        detail = "" if pd.isna(failure) else str(failure)
        status_html = f'<span title="{escape(detail)}">{escape(status.replace("_", " ").capitalize())}</span>'
        css = ' class="highlight"' if name.lower() in {"ensemble", "blend"} else ""
        cells = [f'<th scope="row">{escape(model_name(name))}</th>']
        cells += [f"<td>{value}</td>" for value in values]
        cells += [f"<td>{status_html}</td>"]
        rows.append(f"<tr{css}>{''.join(cells)}</tr>")
    head = "".join(f'<th scope="col">{header}</th>' for header in headers)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def transaction_cost_comparison_note(summary: dict, leaderboard: pd.DataFrame) -> str:
    """Keep the optional cost replay visibly separate from base performance."""
    if leaderboard.empty or "cumulative_return_after_costs" not in leaderboard:
        return ""
    comparison = summary.get("transaction_cost_comparison", {})
    rate = comparison.get("cost_bps") if isinstance(comparison, dict) else None
    if rate is None and "transaction_cost_bps" in leaderboard:
        recorded = pd.to_numeric(leaderboard["transaction_cost_bps"], errors="coerce").dropna().unique()
        rate = recorded[0] if len(recorded) == 1 else None
    try:
        rate = float(rate)
        charged = f"{rate:g} bps per buy and sell" if np.isfinite(rate) and rate >= 0 else "the recorded transaction fees"
    except (TypeError, ValueError):
        charged = "the recorded transaction fees"
    base_rate = summary.get("evaluation", {}).get("cost_bps", summary.get("strategy_config", {}).get("cost_bps"))
    try:
        base_rate = float(base_rate)
        base = f"{base_rate:g} bps" if np.isfinite(base_rate) and base_rate >= 0 else "the base scenario"
    except (TypeError, ValueError):
        base = "the base scenario"
    return ('<p class="comparison-note">Return after costs uses ' + escape(charged)
            + ' in a separate self-financing replay. Return before costs, Sharpe, max drawdown, ending NAV and curves use '
            + escape(base) + '. Blank cost returns indicate an unavailable or invalid replay; hover for details.</p>')


def model_comparison_chart(
    model_daily: pd.DataFrame, directory: Path, *, leaderboard: pd.DataFrame | None = None,
    initial_capital: float | None = None, cost_free: bool = False,
) -> str:
    """Compare actual resolved NAV paths, never substituting a missing result."""
    frame = dates_frame(model_daily)
    if frame.empty or not {"model", "net_nav"}.issubset(frame.columns):
        return empty_chart("No resolved model comparison paths were recorded.")
    valid_names = None
    if leaderboard is not None and not leaderboard.empty and "model" in leaderboard:
        valid_names = set()
        for _, row in leaderboard.iterrows():
            status = str(row.get("status", "resolved")).lower()
            valid = status in {"resolved", "valid", "complete", "recorded", "ok", "nan"}
            valid &= str(row.get("performance_valid", True)).lower() not in {"false", "0"}
            if valid:
                valid_names.add(str(row["model"]))
    palette = ["#437e72", "#7d79a5", "#bf7048", "#538aa0", "#ae8c35", "#9c6489", "#778942", "#557d9d"]
    fig, ax = plt.subplots(figsize=(9.7, 3.5))
    groups = list(frame.groupby("model", sort=False))
    # Draw the blend last so coincident paths still retain the selected forecast.
    groups.sort(key=lambda pair: str(pair[0]).lower() in {"ensemble", "blend"})
    count = 0
    for name, group in groups:
        if valid_names is not None and str(name) not in valid_names:
            continue
        if "performance_valid" in group and group["performance_valid"].astype(str).str.lower().isin(["false", "0"]).any():
            continue
        if "status" in group and not group["status"].astype(str).str.lower().isin(["resolved", "valid", "complete", "recorded", "ok", "nan"]).all():
            continue
        values = finite_series(group, "net_nav")
        if values.empty:
            continue
        is_blend = str(name).lower() in {"ensemble", "blend"}
        is_baseline = str(name).lower() in {"reversal", "momentum", "notebook_baseline"} or "benchmark" in str(name).lower()
        color = INK if is_blend else palette[count % len(palette)]
        ax.plot(values["Date"], values["net_nav"] / 1_000_000,
                color=color, linewidth=2.3 if is_blend else 1.4,
                linestyle="--" if is_baseline else "-", alpha=1 if is_blend else 0.88,
                label=model_name(name), zorder=4 if is_blend else 2)
        count += 1
    if not count:
        plt.close(fig)
        return empty_chart("No complete model comparison paths were recorded.")
    baseline = float(initial_capital) / 1_000_000 if initial_capital is not None else 1.0
    ax.axhline(baseline, color="#bcc9be", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Portfolio NAV ($m)")
    style_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK, columnspacing=1.6, handlelength=2.5)
    format_dates(fig, ax)
    return save_chart(fig, directory, "model_comparison",
                      "Individual model, blend and baseline portfolio NAV paths; costs excluded" if cost_free
                      else "Individual model, blend and baseline portfolio NAV paths after costs")

def render_showcase(
    report_dir: Path, summary: dict, leaderboard: pd.DataFrame, daily: pd.DataFrame,
    top_predictions: pd.DataFrame, feature_importance: pd.DataFrame,
    model_daily: pd.DataFrame | None = None,
) -> Path:
    """Write a model scheme and every recorded strategy comparison."""
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
    if model_daily is None:
        model_path = report_dir / "model_daily.csv"
        model_daily = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
    comparison = model_comparison_chart(model_daily, report_dir, leaderboard=leaderboard,
                                        initial_capital=evaluation.get("initial_capital"), cost_free=cost_free)
    comparison_table = model_results_table(leaderboard)
    cost_comparison_note = transaction_cost_comparison_note(summary, leaderboard)
    external_notes = []
    for name, details in summary.get("external_models", {}).items():
        if not isinstance(details, dict):
            continue
        context = compact(details.get("context_rows"))
        if details.get("status") != "complete":
            explanation = details.get("reason", "A complete test comparison is not available.")
            external_notes.append(f'{escape(model_name(name))}: {escape(explanation)} '
                                  f'Training context: {context} rows. Full-test performance is not measured.')
        else:
            protocol = " Each prediction request contains one signal date only." if details.get("batch_mode") == "single_date" else ""
            if isinstance(details.get("query_protocol"), dict):
                protocol = " Fixed-size queries use frozen real features followed by zero padding; the matching causality diagnostic passed."
            external_notes.append(f'{escape(model_name(name))} is a separate API candidate with a fixed {context}-row training context; '
                                  'it is not included in the frozen blend.' + protocol)
    external_note = ('<p class="comparison-note">' + " ".join(external_notes)
                     + ' <a href="https://github.com/Simon666Z/financial-engineering-club-fall26/blob/SimonResearch/docs/tabpfn.md">TabPFN experiment details</a>.</p>') if external_notes else ""
    training = summary.get("training", {})
    selected = selected_metrics(summary, leaderboard)
    weights = training.get("selected_weights", {}) if isinstance(training, dict) else {}
    candidate_names = list(training.get("candidate_params", {}))
    if not candidate_names:
        candidate_names = ["elastic_net", "xgb_regression", "xgb_ranker", "catboost"]
    if "model" in leaderboard:
        for name in leaderboard["model"].astype(str):
            if name not in candidate_names and name not in {"ensemble", "blend", "reversal", "momentum", "notebook_baseline", *summary.get("external_models", {})} and not any(
                label in name.lower() for label in ["baseline", "benchmark"]):
                candidate_names.append(name)
    candidate_cards = "".join(f'<div class="candidate">{escape(model_name(name))}</div>' for name in candidate_names)
    active = [(name, weight) for name, weight in weights.items() if float(weight) > 0] if isinstance(weights, dict) else []
    if active:
        selected_name = "<br>".join(escape(model_name(name)) for name, _ in active)
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
    is_hold_strategy = str(evaluation.get("strategy_name", "")).startswith("rank_hold") or {"gross_nav", "net_nav"}.issubset(frame.columns)
    top_k = compact(evaluation.get("top_k"))
    entry_k = compact(evaluation.get("entry_k", 20))
    exit_k = compact(evaluation.get("exit_k", 100))
    try:
        long_fraction = float(evaluation.get("long_fraction", 0.5))
    except (TypeError, ValueError):
        long_fraction = 0.5
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
            subject = "This recorded base strategy" if cost_comparison_note else "This recorded strategy"
            outcome = (f"{subject} gains with costs excluded." if result > 0 else
                       f"{subject} loses with costs excluded." if result < 0 else
                       f"{subject} breaks even with costs excluded." if result == 0 else
                       "Portfolio results are not yet available.")
        except (TypeError, ValueError):
            outcome = "Portfolio results are not yet available."
    if evaluation.get("performance_valid") is False:
        outcome = escape(evaluation.get("performance_invalidation_reason") or "Funding rules were breached; accounting NAV is auditable, but performance statistics are invalid.")
    excluded = int(evaluation.get("aggregate_excluded_unresolved_dates", 0) or 0)
    conditional = f" Results use common resolved sessions; {excluded:,} unresolved sessions are excluded." if excluded else ""
    if is_hold_strategy:
        if long_fraction == 1:
            strategy_title = f"Long only<br>Top {entry_k}"
            strategy_detail = f"Hold while top {exit_k}.<br>Invest all free cash."
            strategy_caption = (f"Prior close ranks → next-session adjusted OHLC4 fills. Invest all available free cash across the top {entry_k}, "
                                f"adding to existing positions; retain holdings while in the top {exit_k}. Threshold exits execute next session. "
                                f"Exit cash is available at that fill-day close for following-session entries. {cost} bps commission per side.")
        else:
            strategy_title = f"Top {entry_k} long<br>Bottom {entry_k} short"
            strategy_detail = f"Hold while top / bottom {exit_k}.<br>Holdings can exceed {entry_k} per side."
            strategy_caption = (f"Prior close ranks → next-session adjusted OHLC4 fills. Deploy free cash {number(long_fraction, 0, True)} long / "
                                f"{number(1 - long_fraction, 0, True)} short, adding to top / bottom {entry_k}; hold while top / bottom {exit_k}. "
                                "Exit proceeds become available at the fill-day close. Short proceeds and 100% entry collateral stay segregated. "
                                f"{cost} bps commission per side.")
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
        excluded_caption = "Base results exclude transaction costs." if cost_comparison_note else "Costs excluded for this run."
        strategy_caption = strategy_caption.replace(f"{cost} bps commission per side.", excluded_caption)
        strategy_caption = strategy_caption.replace(f"{cost} bps commission per side; allocations fund buy costs.", excluded_caption)
        proxy_note = proxy_note.replace(" Borrow fee: 0 bps/year.", "")
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FE Club · Model &amp; Strategy</title>
<style>
:root{{--bg:#f1f2eb;--paper:#fbfaf6;--ink:#233a33;--muted:#69756e;--green:#216c58;--line:#dce1d5;--warm:#ac733c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;line-height:1.5}}
main{{max-width:1100px;margin:auto;padding:26px 28px 30px}}header{{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:20px}}.brand{{font-size:13px;font-weight:650}}.brand span{{font-weight:400;color:var(--muted);margin-left:8px}}.period{{font-size:12px;color:var(--muted)}}section{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin-bottom:18px}}h1,h2{{margin:0;font-size:21px;font-weight:600;letter-spacing:-.02em}}.section-head{{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:18px}}.section-head p{{margin:0;color:var(--muted);font-size:12px}}.scheme{{display:grid;grid-template-columns:1fr 16px 1.45fr 16px .88fr 16px 1fr 16px 1.10fr;gap:8px;align-items:stretch}}.node{{border:1px solid var(--line);border-radius:7px;padding:15px 13px;display:flex;flex-direction:column;justify-content:center;min-height:156px}}.node-title{{font-size:12px;font-weight:600;color:var(--muted);margin:0 0 9px}}.node strong{{font-size:16px;font-weight:600;line-height:1.3}}.node p{{font-size:12px;line-height:1.5;color:var(--muted);margin:8px 0 0}}.arrow{{align-self:center;text-align:center;color:#87948b;font-size:22px}}.candidates{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}.candidate{{font-size:12px;line-height:1.4;border:1px solid var(--line);padding:7px 8px;border-radius:5px;background:#f4f5ef;display:flex;align-items:center;min-height:46px}}.selected{{background:#edf3e9;border-color:#b9cdbb}}.selected strong{{color:var(--green)}}.weight{{display:inline-block;align-self:flex-start;font-size:12px;color:var(--green);margin-top:9px;font-weight:600}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:7px;margin-bottom:16px}}.metrics.three{{grid-template-columns:repeat(3,1fr)}}.metric{{padding:14px 18px;border-right:1px solid var(--line)}}.metric:last-child{{border-right:0}}.metric span{{font-size:12px;color:var(--muted);display:block}}.metric strong{{font-weight:500;font-size:28px;letter-spacing:-.03em;display:block;margin-top:4px}}.chart svg{{display:block;width:100%;height:auto}}.chart-empty{{padding:80px 20px;text-align:center;color:var(--muted)}}.capital{{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:-3px 0 16px}}.capital b{{color:var(--ink);font-weight:600}}.proxy{{font-size:12px;color:var(--muted);line-height:1.5;margin:8px 0 0}}.strategy{{font-size:12px;line-height:1.6;color:var(--muted);margin:12px 0 0}}.outcome{{margin:11px 0 0;font-size:13px;color:#7e572e}}footer{{font-size:12px;color:var(--muted);margin:3px 0 0}}.result-label{{font-size:12px;font-weight:600;color:var(--green);margin:-4px 0 12px}}.table-wrap{{overflow-x:auto;margin-top:10px}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}}th:first-child{{text-align:left}}thead th{{font-size:12px;color:var(--muted);font-weight:500}}tbody th{{font-weight:500}}.highlight{{background:#edf3e9}}.highlight th,.highlight td{{font-weight:650}}.comparison-note{{font-size:12px;color:var(--muted);margin:10px 0 0}}
@media(max-width:900px){{.scheme{{grid-template-columns:1fr 1.25fr 1fr;gap:12px}}.arrow{{display:none}}.node{{min-height:130px}}.section-head{{align-items:flex-start;flex-direction:column;gap:4px}}}}
@media(max-width:600px){{main{{padding:18px 14px}}section{{padding:18px 16px}}header{{align-items:flex-start;flex-direction:column;gap:5px}}.scheme{{grid-template-columns:1fr 1fr}}.node{{padding:12px;min-height:125px}}.candidates{{grid-template-columns:1fr}}.node.candidate-node{{grid-row:span 2}}.metrics{{grid-template-columns:1fr 1fr}}.metric:nth-child(2){{border-right:0}}.metric:nth-child(-n+2){{border-bottom:1px solid var(--line)}}.metric strong{{font-size:25px}}.metrics.three .metric:nth-child(2){{border-right:1px solid var(--line)}}.metrics.three .metric:nth-child(-n+2){{border-bottom:0}}.metrics.three .metric{{padding:12px}}}}
</style></head><body><main>
<header><div class="brand">FE CLUB <span>/ SimonResearch</span></div><div class="period">Train {train_period} · Validate {validation_period} · Test {test_period}</div></header>
<section aria-labelledby="scheme-title"><div class="section-head"><h1 id="scheme-title">Model scheme</h1><p>Validation selects forecast settings and blend weights.</p></div>
<div class="scheme" role="list" aria-label="Model and portfolio workflow">
<div class="node" role="listitem"><div class="node-title">Inputs</div><strong>{input_label}</strong><p>{input_caption}</p></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node candidate-node" role="listitem"><div class="node-title">Trained candidates</div><div class="candidates">{candidate_cards}</div><p>Next-day return ranks<br><span>Ranker learns return deciles.</span></p></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node" role="listitem"><div class="node-title">Validation</div><strong>{validation_period}</strong><p>Select model settings and forecast weights.</p></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node selected" role="listitem"><div class="node-title">{selected_label}</div><strong>{selected_name}</strong><span class="weight">{selected_weight}</span></div>
<div class="arrow" aria-hidden="true">→</div>
<div class="node" role="listitem"><div class="node-title">Trading strategy</div><strong>{strategy_title}</strong><p>{strategy_detail}</p></div>
</div></section>
<section aria-labelledby="results-title"><div class="section-head"><h2 id="results-title">{test_period} backtest</h2><p>{sessions} trading sessions · Frozen blend strategy</p></div><p class="result-label">Blend / selected forecast</p>{capital}<div class="{metrics_class}">{''.join(cards)}</div><div class="chart">{comparison}</div>{comparison_table}<p class="comparison-note">Every model and baseline uses the recorded trading policy. Blank metrics mean an incomplete, unresolved or invalid performance result; those paths are omitted from the chart. Solid lines: trained models and blend. Dashed lines: baselines / benchmarks.</p>{cost_comparison_note}{external_note}<p class="strategy">{strategy_caption}{conditional}</p><p class="outcome">{outcome}</p><p class="proxy">{proxy_note}</p></section>
<footer>{footer}</footer>
</main></body></html>'''
    path = report_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


EXTERNAL_CANDIDATE = "tabpfn_3_5"
EXTERNAL_FILES = (
    "reports/tabpfn/status.json", "reports/tabpfn/run.json",
    "reports/tabpfn/batch_causality.json", "reports/tabpfn/test_predictions.parquet",
    "reports/tabpfn/validation_predictions.parquet",
)


def external_candidate_inputs(root: Path) -> dict[str, str]:
    """Fingerprint local optional inputs so cached reports notice status changes."""
    root = Path(root)
    files = set(EXTERNAL_FILES)
    run_path = root / "reports/tabpfn/run.json"
    if run_path.is_file():
        metadata = json.loads(run_path.read_text())
        if metadata.get("batch_mode") == "single_date" and metadata.get("source_report_dir") == "reports/tabpfn-daily":
            files.update("data/processed/tabpfn/" + name for name in SINGLE_DATE_INPUTS)
            for item in metadata.get("requests", []):
                split, day = item.get("split"), item.get("date")
                if split in {"validation", "test"} and isinstance(day, str) and len(day) == 10 and day[:4].isdigit() and day[4] == "-" and day[7] == "-" and day[5:7].isdigit() and day[8:].isdigit():
                    files.add(f"reports/tabpfn-daily/batches/{split}-{day}.parquet")
        if isinstance(metadata.get("query_protocol"), dict) and metadata["query_protocol"].get("name") == "fixed_shape_train_only_padding_v1" and metadata.get("source_report_dir") == "reports/tabpfn-padded":
            files.update("data/processed/tabpfn/" + name for name in SINGLE_DATE_INPUTS)
            files.update({"reports/tabpfn-padded/preflight.json", "reports/tabpfn-padded/batch_causality.json"})
            for item in metadata.get("requests", []):
                batch = item.get("batch", {})
                split, start, stop = batch.get("split"), batch.get("start"), batch.get("stop")
                if split in {"validation", "test", "pilot", "causality_combined", "causality_mutated"} and type(start) is int and type(stop) is int and 0 <= start < stop:
                    base = f"reports/tabpfn-padded/batches/{split}-{start:06d}-{stop:06d}"
                    files.update({base + ".parquet", base + ".json"})
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in sorted(files) if (root / relative).is_file()}


def load_external_candidate(
    root: Path, *, manifest: dict, feature_names: list[str], expected_keys: pd.DataFrame,
) -> tuple[dict, pd.DataFrame | None]:
    """Accept only a verified complete candidate, otherwise retain a status row.

    All provider work is explicit in run_tabpfn.py. Dashboard generation merely
    reads saved artifacts; it never uploads data, charges tokens or changes the
    core forecast table or blend.
    """
    root = Path(root)
    status_path, run_path = root / EXTERNAL_FILES[0], root / EXTERNAL_FILES[1]
    if not status_path.exists() and not run_path.exists():
        return {}, None
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    metadata = json.loads(run_path.read_text()) if run_path.exists() else {}
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        # An active full run supersedes the earlier pilot result, including a
        # resumable quota pause. Neither status can supply test performance.
        active = metadata if metadata.get("mode") == "full" else status
        result = {
            "status": active.get("status", "incomplete"),
            "reason": active.get("reason", "No complete, verified TabPFN test run is available."),
            "context_rows": active.get("context_rows", status.get("context_rows")),
            "model_path": active.get("model_path", status.get("model_path")),
            "n_estimators": active.get("n_estimators", status.get("n_estimators")),
            "batch_mode": active.get("batch_mode"), "query_protocol": active.get("query_protocol"),
            "completed_requests": active.get("completed_requests", 0),
            "planned_requests": active.get("planned_requests"), "performance_valid": False,
        }
        # A status-only file cannot authorize a complete portfolio estimate.
        if result["status"] == "complete":
            result["status"] = "incomplete"
        return result, None

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("Invalid external TabPFN candidate: " + message)

    require(metadata.get("candidate_key") == EXTERNAL_CANDIDATE, "wrong candidate key")
    require(metadata.get("alpha_data_sha256") == manifest["sha256"], "alpha snapshot mismatch")
    require(metadata.get("raw_snapshot_sha256") == manifest["raw_snapshot_sha256"], "raw snapshot mismatch")
    require(metadata.get("feature_names") == list(feature_names), "feature schema mismatch")
    context = metadata.get("training_context", {})
    require(context.get("feature_names") == list(feature_names), "context schema mismatch")
    require(context.get("transforms_before_sampling") is True, "unverified feature transform timing")
    require(context.get("train_end") == metadata.get("train_end") and context.get("train_label_end") == metadata.get("train_label_end"), "context training boundaries mismatch")
    require(context.get("alpha_data_sha256") == manifest["sha256"] and context.get("raw_snapshot_sha256") == manifest["raw_snapshot_sha256"], "context snapshot mismatch")
    require(pd.Timestamp(metadata.get("train_end")) <= pd.Timestamp("2022-12-29"), "training extends into validation")
    require(pd.Timestamp(metadata.get("train_label_end")) <= pd.Timestamp("2022-12-30"), "training label crosses its boundary")
    require(metadata.get("test_labels_used") is False and metadata.get("blend_changed") is False, "test-label or blend policy violated")
    require(metadata.get("model_path") == "v3.5_default" and metadata.get("n_estimators") == 4, "model configuration mismatch")
    candidate_config = metadata.get("model_config", {})
    require(candidate_config.get("fit_mode") in {"fit_preprocessors", "fit_with_cache"}
            and candidate_config == {"model_path": "v3.5_default", "n_estimators": 4, "random_state": 42,
                                     "fit_mode": candidate_config.get("fit_mode")}, "unverified model configuration")
    require(metadata.get("context_rows") == context.get("train_rows") and int(metadata.get("context_rows", 0)) > 0, "context count mismatch")
    for relative, expected in context.get("prepared_alpha_sha256", {}).items():
        require(relative in {"data/processed/alpha_model_data.parquet", "data/processed/alpha_panel.parquet", "data/processed/alpha_data_manifest.json"}, "unexpected prepared artifact")
        path = root / relative
        require(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, "prepared artifact changed")
    require(len(context.get("prepared_alpha_sha256", {})) == 3, "missing prepared checksums")
    single_date = metadata.get("batch_mode") == "single_date"
    if single_date:
        require(metadata.get("query_protocol") == "single_date_query", "unknown single-date query protocol")
    else:
        require(metadata.get("batch_mode") in {None, "multi_date"}, "unknown batch mode")
        proof_path = root / "reports/tabpfn/batch_causality.json"
        require(proof_path.exists(), "missing batch-independence proof")
        require(hashlib.sha256(proof_path.read_bytes()).hexdigest() == metadata.get("batch_independence_proof_sha256"), "proof checksum mismatch")
        proof = json.loads(proof_path.read_text())
        require(proof == metadata.get("batch_independence_proof"), "proof metadata mismatch")
        require(proof.get("status") == "passed" and proof.get("ranking_identical") is True, "batch-independence diagnostic failed")
        require(proof.get("input_fingerprint") == metadata.get("input_fingerprint"), "proof uses a different input context")
        require(proof.get("fitted_train_set_id") == metadata.get("fitted_train_set_id"), "proof uses a different provider fit")
        require(proof.get("feature_names") == list(feature_names) and proof.get("model_config") == metadata.get("model_config"), "proof uses a different schema or model")
        require(proof.get("query_protocol") == metadata.get("query_protocol"), "proof uses a different query protocol")
        try:
            tolerance = float(proof["tolerance"])
            differences = [float(proof[key]) for key in ["max_abs_date_alone", "max_abs_future_mutation"]]
            require(0 <= tolerance <= 1e-6 and all(np.isfinite(value) and 0 <= value <= tolerance for value in differences), "proof tolerance or differences invalid")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid external TabPFN candidate: malformed numeric proof") from error
    hashes = metadata.get("output_sha256", {})
    for name in ["test_predictions.parquet", "validation_predictions.parquet"]:
        path = root / "reports/tabpfn" / name
        require(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == hashes.get(name), "forecast checksum mismatch")
    forecast = pd.read_parquet(root / "reports/tabpfn/test_predictions.parquet")
    require({"Date", "Ticker", "prediction"}.issubset(forecast), "missing forecast columns")
    forecast = forecast[["Date", "Ticker", "prediction"]].copy()
    forecast["Date"] = pd.to_datetime(forecast["Date"], errors="coerce")
    forecast["prediction"] = pd.to_numeric(forecast["prediction"], errors="coerce")
    require(not forecast[["Date", "Ticker"]].isna().any().any() and not forecast.duplicated(["Date", "Ticker"]).any(), "invalid or duplicate forecast keys")
    require(np.isfinite(forecast["prediction"]).all(), "nonfinite forecasts")
    keys = expected_keys[["Date", "Ticker"]].copy()
    keys["Date"] = pd.to_datetime(keys["Date"])
    require(not keys.duplicated(["Date", "Ticker"]).any(), "duplicate expected test keys")
    coverage = keys.merge(forecast, on=["Date", "Ticker"], how="outer", indicator=True, validate="one_to_one")
    require(len(coverage) == len(keys) and coverage["_merge"].eq("both").all(), "incomplete or extra test coverage")
    if single_date:
        validate_single_date_requests(root, metadata, feature_names=list(feature_names), forecast=forecast)
    elif isinstance(metadata.get("query_protocol"), dict):
        validate_padded_requests(root, metadata, feature_names=list(feature_names), forecast=forecast)
    elif metadata.get("query_protocol") not in {None, "multi_date_query"}:
        raise ValueError("Invalid external TabPFN candidate: unknown query protocol")
    return {
        "status": "complete", "context_rows": metadata["context_rows"],
        "model_path": metadata["model_path"], "n_estimators": metadata["n_estimators"],
        "alpha_data_sha256": metadata["alpha_data_sha256"], "raw_snapshot_sha256": metadata["raw_snapshot_sha256"],
        "input_fingerprint": metadata["input_fingerprint"], "train_end": metadata["train_end"],
        "train_label_end": metadata["train_label_end"], "performance_valid": True,
        "blend_inclusion": False, "batch_mode": metadata.get("batch_mode", "multi_date"),
        "query_protocol": metadata.get("query_protocol", "multi_date_query"),
        "fit_mode": metadata.get("model_config", {}).get("fit_mode"),
        "completed_requests": len(metadata.get("requests", [])),
    }, forecast.rename(columns={"prediction": EXTERNAL_CANDIDATE})


SINGLE_DATE_INPUTS = (
    "manifest.json", "train_features.parquet", "train_labels.npy", "train_keys.parquet",
    "validation_features.parquet", "validation_keys.parquet", "test_features.parquet", "test_keys.parquet",
)


def parquet_digest(frame: pd.DataFrame) -> str:
    """Match the adapter's exact target-free upload bytes, without calling it."""
    output = io.BytesIO()
    frame.to_parquet(output, index=False, compression="zstd")
    return hashlib.sha256(output.getvalue()).hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_single_date_requests(root: Path, metadata: dict, *, feature_names: list[str], forecast: pd.DataFrame) -> None:
    """Prove saved full coverage used one date per query and a frozen train fit.

    A failed multi-date equivalence diagnostic remains failed. Date isolation
    supplies a different causal contract: no later query rows reach a forecast.
    """
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("Invalid external TabPFN single-date candidate: " + message)

    config = metadata.get("model_config", {})
    require(config == {"model_path": "v3.5_default", "n_estimators": 4, "random_state": 42,
                       "fit_mode": config.get("fit_mode")}, "model config differs from the fixed candidate")
    require(config.get("fit_mode") in {"fit_preprocessors", "fit_with_cache"}, "unverified fit mode")
    expected_run = canonical_digest({"input_fingerprint": metadata.get("input_fingerprint"), "model_config": config,
                                     "query_protocol": "single_date_query"})
    require(metadata.get("run_fingerprint") == expected_run, "run fingerprint mismatch")
    require(metadata.get("source_report_dir") == "reports/tabpfn-daily", "unexpected source report directory")
    context = metadata["training_context"]
    prepared = root / "data/processed/tabpfn"
    saved_manifest = json.loads((prepared / "manifest.json").read_text())
    require(saved_manifest == context, "training context manifest changed")
    input_hashes = {}
    for name in SINGLE_DATE_INPUTS:
        if name == "manifest.json":
            continue
        path = prepared / name
        require(path.is_file(), "missing prepared input " + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == context.get("sha256", {}).get(name), "prepared input changed: " + name)
        input_hashes[name] = actual
    # The existing input fingerprint retains the standard fit selector; the
    # separate run fingerprint records cache execution without altering data.
    fingerprint_manifest = dict(context)
    fingerprint_manifest["sha256"] = {name: value for name, value in context.get("sha256", {}).items() if name in input_hashes}
    standard_config = {"model_path": "v3.5_default", "n_estimators": 4, "random_state": 42, "fit_mode": "fit_preprocessors"}
    expected_input = canonical_digest({"manifest": fingerprint_manifest, "sha256": input_hashes, "model_config": standard_config})
    require(metadata.get("input_fingerprint") == expected_input, "frozen input fingerprint mismatch")
    train_keys = pd.read_parquet(prepared / "train_keys.parquet")
    train_dates = pd.to_datetime(train_keys["Date"], errors="coerce")
    require(len(train_keys) == metadata["context_rows"] and train_dates.notna().all()
            and train_dates.le(pd.Timestamp(metadata["train_end"])).all()
            and not train_keys.duplicated(["Date", "Ticker"]).any(), "training keys cross the fixed boundary")
    requests = metadata.get("requests")
    require(isinstance(requests, list) and bool(requests), "missing completed date requests")
    seen = set()
    split_inputs = {}
    aggregates = {"test": forecast}
    for split in ["validation", "test"]:
        keys = pd.read_parquet(prepared / f"{split}_keys.parquet").reset_index(drop=True)
        require(list(keys.columns) == ["Date", "Ticker"] and not keys.duplicated(["Date", "Ticker"]).any(), "invalid prepared keys")
        keys["Date"] = pd.to_datetime(keys["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        keys["Ticker"] = keys["Ticker"].astype(str)
        first, last = ("2023-01-01", "2023-12-31") if split == "validation" else ("2024-01-01", "2025-12-31")
        require(keys["Date"].notna().all() and keys["Date"].between(first, last).all(), "prepared split date boundary violation")
        require(keys.equals(keys.sort_values(["Date", "Ticker"]).reset_index(drop=True)), "unsorted prepared keys")
        features = pd.read_parquet(prepared / f"{split}_features.parquet").reset_index(drop=True)
        require(list(features.columns) == feature_names and len(features) == len(keys)
                and np.isfinite(features.to_numpy(dtype=float)).all(), "prepared split feature mismatch")
        split_inputs[split] = (keys, features)
        if split == "validation":
            aggregates[split] = pd.read_parquet(root / "reports/tabpfn/validation_predictions.parquet")
        aggregate = aggregates[split]
        require(list(aggregate.columns) == ["Date", "Ticker", "prediction"], "aggregate forecast schema mismatch")
        require(aggregate[["Date", "Ticker"]].reset_index(drop=True).equals(keys)
                and np.isfinite(aggregate["prediction"]).all(), "aggregate split coverage mismatch")
    expected_dates = {(split, day.strftime("%Y-%m-%d")) for split, (keys, _) in split_inputs.items() for day in keys["Date"].drop_duplicates()}
    for item in requests:
        split, day = item.get("split"), item.get("date")
        require((split, day) in expected_dates and (split, day) not in seen, "duplicate or unexpected date request")
        seen.add((split, day))
        require(item.get("status") == "completed", "request did not complete")
        require(item.get("input_fingerprint") == metadata["input_fingerprint"] and item.get("run_fingerprint") == expected_run
                and item.get("fitted_train_set_id") == metadata.get("fitted_train_set_id")
                and item.get("model_config") == config, "request uses different data, fit or settings")
        keys, features = split_inputs[split]
        mask = keys["Date"].eq(pd.Timestamp(day))
        indices = np.flatnonzero(mask.to_numpy())
        selected_keys = keys.loc[mask].reset_index(drop=True)
        selected_features = features.loc[mask].reset_index(drop=True)
        require(item.get("first_date") == day and item.get("last_date") == day and item.get("rows") == len(selected_keys), "multi-date or partial-date request")
        batch = item.get("batch", {})
        require(batch == {"split": split, "start": int(indices[0]), "stop": int(indices[-1]) + 1,
                          "first_date": day, "last_date": day, "rows": len(selected_keys)}, "batch slice does not match its sole date")
        require(item.get("query_keys_sha256") == parquet_digest(selected_keys)
                and item.get("query_features_sha256") == parquet_digest(selected_features), "query contains different keys or future features")
        relative = f"batches/{split}-{day}.parquet"
        require(item.get("output_relative_path") == relative, "unsafe or wrong batch output path")
        path = root / metadata["source_report_dir"] / relative
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item.get("predictions_sha256"), "batch forecast checksum mismatch")
        batch_forecast = pd.read_parquet(path)
        require(list(batch_forecast.columns) == ["Date", "Ticker", "prediction"]
                and batch_forecast[["Date", "Ticker"]].equals(selected_keys)
                and np.isfinite(batch_forecast["prediction"]).all(), "batch forecast keys or values invalid")
        expected_values = aggregates[split].loc[mask, "prediction"].to_numpy()
        require(np.array_equal(batch_forecast["prediction"].to_numpy(), expected_values), "aggregate differs from its completed request")
    require(seen == expected_dates, "incomplete validation or test date coverage")


PADDED_QUERY_PROTOCOL = {
    "name": "fixed_shape_train_only_padding_v1", "query_rows": 10000,
    "dtype": "float32", "padding_value": 0.0, "padding_placement": "tail",
    "feature_transform_timing": "complete real date cross-sections before packing",
    "ordering": "Date,Ticker", "discard_dummy_outputs": True,
    "block_policy": "greedy chronological whole dates; start depends only on past/current row counts",
}


def load_padded_frozen_inputs(root: Path, context: dict, feature_names: list[str]) -> tuple[str, dict]:
    """Load target-free query inputs and reconstruct their recorded fingerprint."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("Invalid external TabPFN padded candidate: " + message)
    prepared = root / "data/processed/tabpfn"
    require(json.loads((prepared / "manifest.json").read_text()) == context, "context manifest changed")
    hashes = {}
    for name in SINGLE_DATE_INPUTS:
        if name == "manifest.json":
            continue
        path = prepared / name
        require(path.is_file(), "missing prepared input " + name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(hashes[name] == context.get("sha256", {}).get(name), "prepared input changed: " + name)
    fingerprint_manifest = dict(context)
    fingerprint_manifest["sha256"] = {name: value for name, value in context.get("sha256", {}).items() if name in hashes}
    standard_config = {"model_path": "v3.5_default", "n_estimators": 4, "random_state": 42, "fit_mode": "fit_preprocessors"}
    fingerprint = canonical_digest({"manifest": fingerprint_manifest, "sha256": hashes, "model_config": standard_config})
    train_keys = pd.read_parquet(prepared / "train_keys.parquet")
    train_dates = pd.to_datetime(train_keys["Date"], errors="coerce")
    require(len(train_keys) == context["train_rows"] and train_dates.notna().all()
            and train_dates.le(pd.Timestamp(context["train_end"])).all()
            and not train_keys.duplicated(["Date", "Ticker"]).any(), "training key boundary violation")
    inputs = {}
    for split in ["validation", "test"]:
        keys = pd.read_parquet(prepared / f"{split}_keys.parquet").reset_index(drop=True)
        require(list(keys.columns) == ["Date", "Ticker"] and not keys.duplicated(["Date", "Ticker"]).any(), "invalid prepared keys")
        keys["Date"] = pd.to_datetime(keys["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        keys["Ticker"] = keys["Ticker"].astype(str)
        first, last = ("2023-01-01", "2023-12-31") if split == "validation" else ("2024-01-01", "2025-12-31")
        require(not keys.empty and keys["Date"].notna().all() and keys["Date"].between(first, last).all()
                and keys.equals(keys.sort_values(["Date", "Ticker"]).reset_index(drop=True)), "prepared split key boundary/order violation")
        features = pd.read_parquet(prepared / f"{split}_features.parquet").reset_index(drop=True)
        require(list(features.columns) == feature_names and len(features) == len(keys)
                and np.isfinite(features.to_numpy(dtype=float)).all(), "prepared feature schema or values mismatch")
        inputs[split] = (keys, features)
    return fingerprint, inputs


def padded_batch_plan(keys: pd.DataFrame, split: str) -> list[dict]:
    """Reproduce the fixed chronological whole-date packing policy."""
    groups = [(int(offsets.min()), int(offsets.max()) + 1) for offsets in keys.groupby("Date", sort=False).indices.values()]
    batches = []
    start, stop = groups[0][0], groups[0][0]
    for begin, end in groups:
        if end - begin > 10000:
            raise ValueError("Invalid external TabPFN padded candidate: date cross-section exceeds fixed shape")
        if end - start > 10000:
            batches.append({"split": split, "start": start, "stop": stop, "rows": stop - start,
                            "first_date": str(keys.iloc[start]["Date"].date()), "last_date": str(keys.iloc[stop - 1]["Date"].date())})
            start = begin
        stop = end
    batches.append({"split": split, "start": start, "stop": stop, "rows": stop - start,
                    "first_date": str(keys.iloc[start]["Date"].date()), "last_date": str(keys.iloc[stop - 1]["Date"].date())})
    return batches


def validate_padded_requests(root: Path, metadata: dict, *, feature_names: list[str], forecast: pd.DataFrame) -> None:
    """Accept only the preregistered fixed shape that passed its own causal gate."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("Invalid external TabPFN padded candidate: " + message)
    protocol = metadata.get("query_protocol")
    require(protocol == PADDED_QUERY_PROTOCOL, "fixed query protocol changed")
    require(metadata.get("source_report_dir") == "reports/tabpfn-padded", "unexpected source report directory")
    require(metadata["model_config"]["fit_mode"] == "fit_with_cache", "fixed protocol requires its cached fit")
    proof = metadata["batch_independence_proof"]
    require(proof.get("query_protocol") == protocol and proof.get("query_rows_constant") == 10000,
            "proof is not bound to the exact fixed query shape")
    prepared_fingerprint, inputs = load_padded_frozen_inputs(root, metadata["training_context"], feature_names)
    expected_fingerprint = canonical_digest({"prepared_fingerprint": prepared_fingerprint, "model_config": metadata["model_config"], "query_protocol": protocol})
    require(metadata.get("input_fingerprint") == expected_fingerprint, "input/config/protocol fingerprint mismatch")
    require(proof.get("input_sha256") == {name: value for name, value in metadata["training_context"]["sha256"].items() if name in SINGLE_DATE_INPUTS and name != "manifest.json"}, "proof source inputs mismatch")
    source = root / metadata["source_report_dir"]
    source_proof = source / "batch_causality.json"
    require(source_proof.is_file() and hashlib.sha256(source_proof.read_bytes()).hexdigest() == metadata["batch_independence_proof_sha256"], "source proof checksum mismatch")
    preflight_path = source / "preflight.json"
    require(preflight_path.is_file(), "missing pre-inference protocol record")
    preflight = json.loads(preflight_path.read_text())
    require(preflight.get("query_protocol") == protocol and preflight.get("model_config") == metadata["model_config"]
            and preflight.get("input_fingerprint") == expected_fingerprint, "preregistered protocol/data/model mismatch")
    registered_at = pd.to_datetime(preflight.get("generated_at"), utc=True, errors="coerce")
    require(pd.notna(registered_at), "invalid protocol registration timestamp")
    requests = metadata.get("requests")
    require(isinstance(requests, list) and bool(requests), "missing completed query journal")
    aggregates = {"test": forecast, "validation": pd.read_parquet(root / "reports/tabpfn/validation_predictions.parquet")}
    expected = {}
    for split, (keys, _) in inputs.items():
        aggregate = aggregates[split]
        require(list(aggregate.columns) == ["Date", "Ticker", "prediction"] and aggregate[["Date", "Ticker"]].reset_index(drop=True).equals(keys)
                and np.isfinite(aggregate["prediction"]).all(), "aggregate split coverage mismatch")
        for batch in padded_batch_plan(keys, split):
            expected[canonical_digest(batch)] = batch
    require(preflight.get("full_batches") == list(expected.values()), "recorded full-batch plan changed")
    normal = [item for item in requests if item.get("batch", {}).get("split") in {"validation", "test"}]
    seen = set()
    for item in normal:
        batch = item.get("batch", {})
        identifier = canonical_digest(batch)
        require(identifier in expected and identifier not in seen and item.get("batch_id") == identifier,
                "duplicate, unexpected or partial-date query batch")
        seen.add(identifier)
        require(item.get("status") == "completed" and item.get("fitted_train_set_id") == metadata.get("fitted_train_set_id"), "query incomplete or uses another fit")
        require(item.get("query_protocol") == protocol and item.get("query_rows") == 10000 and item.get("real_rows") == batch["rows"], "query shape or real-row count changed")
        started_at = pd.to_datetime(item.get("started_at"), utc=True, errors="coerce")
        require(pd.notna(started_at) and started_at >= registered_at, "query predates its recorded protocol")
        keys, features = inputs[batch["split"]]
        selected = features.iloc[batch["start"]:batch["stop"]].reset_index(drop=True).astype(np.float32)
        zeros = pd.DataFrame(np.zeros((10000 - len(selected), len(feature_names)), dtype=np.float32), columns=feature_names)
        padded = pd.concat([selected, zeros], ignore_index=True)
        uploaded = canonical_digest({"parquet_sha256": parquet_digest(padded), "columns": feature_names, "rows": 10000})
        require(item.get("uploaded_query_sha256") == uploaded, "uploaded matrix differs from frozen features/zero padding")
        stem = f"{batch['split']}-{batch['start']:06d}-{batch['stop']:06d}"
        prediction_path, response_path = source / "batches" / (stem + ".parquet"), source / "batches" / (stem + ".json")
        require(prediction_path.is_file() and hashlib.sha256(prediction_path.read_bytes()).hexdigest() == item.get("predictions_sha256"), "batch forecast checksum mismatch")
        require(response_path.is_file() and hashlib.sha256(response_path.read_bytes()).hexdigest() == item.get("response_metadata_sha256"), "batch response checksum mismatch")
        response = json.loads(response_path.read_text())
        require(response.get("batch") == batch and response.get("query_protocol") == protocol and response.get("input_fingerprint") == expected_fingerprint
                and response.get("query_rows") == 10000 and response.get("real_rows") == batch["rows"] and response.get("uploaded_query_sha256") == uploaded,
                "batch response provenance mismatch")
        provider = response.get("provider_metadata", {})
        require(provider.get("test_set_num_rows") == 10000 and provider.get("test_set_num_cols") == len(feature_names)
                and provider.get("n_estimators") == 4 and provider.get("billing_model_version") == "v3.5" and provider.get("task") == "regression",
                "provider query shape/model/task mismatch")
        selected_keys = keys.iloc[batch["start"]:batch["stop"]].reset_index(drop=True)
        piece = pd.read_parquet(prediction_path)
        require(list(piece.columns) == ["Date", "Ticker", "prediction"] and piece[["Date", "Ticker"]].equals(selected_keys)
                and np.isfinite(piece["prediction"]).all(), "batch predictions do not cover the real keys")
        expected_values = aggregates[batch["split"]].iloc[batch["start"]:batch["stop"]]["prediction"].to_numpy()
        require(np.array_equal(piece["prediction"].to_numpy(), expected_values), "aggregate differs from its saved batch predictions")
    require(seen == set(expected), "incomplete validation/test batch coverage")
    diagnostics = [item for item in requests if item not in normal]
    diagnostic_splits = [item.get("batch", {}).get("split") for item in diagnostics]
    require(sorted(diagnostic_splits) == ["causality_combined", "causality_mutated", "pilot"], "missing or unexpected diagnostic queries")
    for item in diagnostics:
        require(item.get("status") == "completed" and item.get("fitted_train_set_id") == metadata.get("fitted_train_set_id")
                and item.get("query_protocol") == protocol and item.get("query_rows") == 10000,
                "diagnostic query uses another fit/protocol/shape or did not complete")
