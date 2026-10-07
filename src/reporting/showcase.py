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


def render_showcase(
    report_dir: Path, summary: dict, leaderboard: pd.DataFrame, daily: pd.DataFrame,
    top_predictions: pd.DataFrame, feature_importance: pd.DataFrame,
) -> Path:
    """Write charts and a self-contained index.html using actual experiment data."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = dates_frame(daily)
    portfolio_frame = resolved_portfolio_frame(frame)
    net = equity_chart(portfolio_frame, report_dir, "net")
    gross = equity_chart(portfolio_frame, report_dir, "gross")
    drawdown = drawdown_chart(portfolio_frame, report_dir)
    ic = ic_chart(frame, report_dir)
    importance = importance_chart(feature_importance, report_dir)
    metrics = selected_metrics(summary, leaderboard)
    dataset, training = summary.get("dataset", {}), summary.get("training", {})
    evaluation, splits = summary.get("evaluation", {}), summary.get("splits", {})
    cards = []
    for key, title, note, percent, decimals in [
        ("rank_ic", "Rank IC", "Mean daily forecast–return rank correlation", False, 3),
        ("cumulative_net_return", "Net cumulative return", "After the recorded turnover cost assumption", True, 1),
        ("sharpe_net", "Net Sharpe", "Annualized daily mean divided by volatility", False, 2),
        ("max_drawdown_net", "Maximum drawdown", "Largest fall from the net equity high-water mark", True, 1),
    ]:
        cards.append(f'<article class="metric"><p>{title}</p><strong>{number(metrics.get(key), decimals, percent)}</strong><span>{note}</span></article>')
    split_cards = []
    for split, fallback, purpose in [("train", "2016–2022", "Fit candidate models"), ("validation", "2023", "Select settings and ensemble weights"), ("test", "2024–2025", "Measure the selected experiment")]:
        detail = splits.get(split, {})
        span = f'{date(detail.get("start"))} – {date(detail.get("end"))}' if detail.get("start") and detail.get("end") else fallback
        rows = f' · {compact(detail["rows"])} rows' if "rows" in detail else ""
        split_cards.append(f'<article class="split"><span>{split}</span><strong>{span}</strong><p>{purpose}{rows}</p></article>')
    weights = training.get("selected_weights", {}) if isinstance(training, dict) else {}
    active_components = [name for name, weight in weights.items() if float(weight) > 0] if isinstance(weights, dict) else []
    selection_text = "a validation-selected single model" if len(active_components) == 1 else "a validation-selected model blend"
    weights_html = "".join(f'<span class="chip">{escape(name)} <b>{number(weight, 1, True)}</b></span>' for name, weight in weights.items()) if isinstance(weights, dict) else ""
    board = table(leaderboard, [
        ("model", "Model", "text"), ("rank_ic", "Rank IC", "number"),
        ("cumulative_gross_return", "Gross return", "percent"), ("cumulative_net_return", "Net return", "percent"), ("sharpe_net", "Net Sharpe", "number"),
        ("max_drawdown_net", "Max drawdown", "percent"), ("rmse_if_return_prediction", "Return RMSE", "precise"),
    ], highlight=True)
    predictions = top_predictions.copy()
    prediction_date = "Exported prediction snapshot"
    if "Date" in predictions:
        dates = pd.to_datetime(predictions["Date"], errors="coerce")
        if dates.notna().any():
            predictions = predictions.loc[dates.eq(dates.max())]
            prediction_date = f"Final exported signal date · {date(dates.max())}"
    if "rank" in predictions:
        predictions = predictions.sort_values("rank")
    elif "score" in predictions:
        predictions = predictions.sort_values("score", ascending=False)
    prediction_table = table(predictions.head(10), [
        ("rank", "Rank", "integer"), ("Ticker", "Ticker", "text"), ("prediction", "Predicted return", "percent"),
        ("score", "Rank score", "number"), ("disagreement", "Disagreement", "number"),
    ])
    limitations = summary.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    if not limitations:
        limitations = [
            "Current-universe historical data is exploratory and may contain survivorship bias.",
            "The demo does not establish live profitability or a publishable research contribution.",
        ]
    limitations_html = "".join(f"<li>{escape(item)}</li>" for item in limitations)
    metadata = escape(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    sha = escape(dataset.get("sha256", dataset.get("sha", "unavailable")))
    cost = number(evaluation.get("cost_bps"), 1) if isinstance(evaluation, dict) else "—"
    top_k = compact(evaluation.get("top_k")) if isinstance(evaluation, dict) else "—"
    unresolved = compact(evaluation.get("unresolved_trading_dates")) if isinstance(evaluation, dict) else "—"
    excluded = int(evaluation.get("aggregate_excluded_unresolved_dates", len(frame) - len(portfolio_frame)) or 0) if isinstance(evaluation, dict) else len(frame) - len(portfolio_frame)
    conditional_note = f'<p class="conditional-note">{excluded:,} session(s) lack resolved prices for one or both portfolios. Return, Sharpe, drawdown and equity use common resolved sessions only and are conditional results, not full-period performance.</p>' if excluded else ""
    output = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FE Club · SimonResearch Showcase</title>
<style>
:root{{--bg:#f1f2eb;--paper:#fbfaf6;--ink:#233a33;--muted:#758078;--green:#216c58;--line:#dce1d5;--warm:#ac733c}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;line-height:1.5}}
a{{color:inherit}}header{{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:25px 0;border-bottom:1px solid var(--line)}}main{{max-width:1180px;margin:auto;padding:0 38px 44px}}.brand{{font-weight:750;letter-spacing:.04em;font-size:13px}}.brand span{{color:var(--muted);font-weight:450;margin-left:10px}}nav{{display:flex;gap:24px}}nav a{{font-size:12px;text-decoration:none;color:var(--muted)}}nav a:hover{{color:var(--green)}}.hero{{padding:48px 0 30px;display:grid;grid-template-columns:1.35fr .65fr;gap:45px;align-items:end}}.eyebrow{{text-transform:uppercase;font-size:10px;font-weight:750;letter-spacing:.18em;color:var(--green);margin:0 0 17px}}h1{{font-family:Georgia,"Times New Roman",serif;font-size:clamp(38px,5.2vw,66px);font-weight:500;letter-spacing:-.055em;line-height:1.05;margin:0 0 18px}}h1 em{{color:var(--green);font-weight:400}}.intro{{max-width:640px;color:#617168;font-size:15px;line-height:1.7;margin:0}}.snapshot{{border-left:1px solid var(--line);padding-left:30px}}.snapshot strong{{font-size:26px;font-weight:500}}.snapshot p{{margin:4px 0 18px;font-size:11px;color:var(--muted)}}.snapshot .date-span{{font-size:13px;line-height:1.6;color:var(--ink)}}.runline{{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:var(--muted);padding-bottom:26px;overflow-wrap:anywhere}}.status{{color:var(--green)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:11px;background:var(--paper);overflow:hidden;margin-bottom:35px}}.metric{{padding:23px 22px;border-right:1px solid var(--line)}}.metric:last-child{{border-right:0}}.metric p{{font-size:11px;margin:0 0 9px;color:var(--muted)}}.metric strong{{display:block;font-size:34px;font-weight:500;letter-spacing:-.04em;margin-bottom:9px}}.metric span{{display:block;font-size:10px;line-height:1.5;color:var(--muted);max-width:190px}}.panel{{background:var(--paper);border:1px solid var(--line);border-radius:11px;padding:25px;margin-bottom:22px}}.panel-head{{display:flex;justify-content:space-between;gap:15px;align-items:start;margin-bottom:18px}}h2{{font-size:19px;font-weight:550;letter-spacing:-.025em;line-height:1.3;margin:0 0 7px}}h3{{font-size:13px;font-weight:650;margin:0 0 7px}}.muted,.panel-head p{{color:var(--muted);font-size:11px;margin:0;line-height:1.6}}.segments{{display:flex;padding:3px;background:#edf0e8;border:1px solid var(--line);border-radius:7px;gap:3px}}button{{font:inherit;cursor:pointer}}.segments button{{border:0;background:transparent;border-radius:5px;padding:6px 12px;font-size:10px;color:var(--muted)}}.segments button[aria-pressed="true"]{{background:var(--paper);color:var(--green);box-shadow:0 1px 3px #253c3320}}button:focus-visible,a:focus-visible,summary:focus-visible{{outline:2px solid var(--green);outline-offset:3px}}.chart svg{{width:100%;height:auto;display:block}}.chart-empty{{display:flex;align-items:center;justify-content:center;min-height:230px;background:#f0f2eb;border-radius:7px;color:var(--muted);font-size:12px;padding:25px}}.verdict{{border:1px solid #dacdbb;background:#f4eee4;padding:24px;border-radius:11px;margin:-13px 0 28px}}.verdict .eyebrow{{color:var(--warm);margin-bottom:11px}}.verdict-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:25px}}.verdict-grid p{{font-size:11px;color:#766c5d;line-height:1.7;margin:7px 0}}.verdict-grid b{{color:#65543e;font-weight:600}}.conditional-note{{color:#785323;background:#f4eadb;padding:12px 15px;border-radius:6px;font-size:11px;line-height:1.6}}.chart-note{{color:var(--muted);font-size:10px;line-height:1.6;margin:14px 0 0}}.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}.two-col .panel{{min-width:0}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:11px;white-space:nowrap}}th,td{{padding:12px 11px;border-bottom:1px solid #e6e9e0;text-align:right}}th{{color:var(--muted);font-weight:500;font-size:10px}}th:first-child,td:first-child{{text-align:left}}tbody tr:last-child td{{border-bottom:0}}tbody tr.highlight{{background:#eef3ea;font-weight:650;color:var(--green)}}.splits{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:18px 0 27px}}.split{{padding-left:15px;border-left:2px solid #c5d6c7}}.split:last-child{{border-color:var(--warm)}}.split span{{display:block;text-transform:uppercase;color:var(--muted);letter-spacing:.12em;font-size:9px;margin-bottom:6px}}.split strong{{font-size:12px;font-weight:550}}.split p{{font-size:10px;line-height:1.6;color:var(--muted);margin:7px 0 0}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:20px}}.flow-item{{border:1px solid var(--line);border-radius:7px;padding:15px}}.flow-item span{{color:var(--warm);font-size:10px;display:block;margin-bottom:8px}}.flow-item p{{font-size:10px;color:var(--muted);line-height:1.6;margin:0}}.weights{{display:flex;gap:8px;flex-wrap:wrap}}.chip{{font-size:10px;color:var(--muted);border:1px solid var(--line);border-radius:20px;padding:5px 11px}}.chip b{{font-weight:600;color:var(--green);margin-left:6px}}.scope{{background:#e9ede2;border:1px solid #d3dccb;border-radius:11px;padding:26px;margin-bottom:22px}}.scope ul{{margin:12px 0 0;padding-left:18px}}.scope li{{font-size:11px;line-height:1.7;color:#63715f;margin:5px 0}}details{{border-top:1px solid var(--line);padding-top:18px}}summary{{cursor:pointer;font-size:11px;color:var(--muted)}}pre{{font-family:ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#e9ece4;padding:18px;border-radius:8px;font-size:10px;max-height:420px;overflow:auto}}footer{{display:flex;justify-content:space-between;gap:20px;font-size:10px;color:var(--muted);padding-top:24px;border-top:1px solid var(--line)}}[hidden]{{display:none!important}}
@media(max-width:800px){{main{{padding:0 20px 30px}}.hero{{grid-template-columns:1fr;gap:25px;padding-top:32px}}.snapshot{{border-left:0;border-top:1px solid var(--line);padding:18px 0 0;display:flex;gap:30px;align-items:start}}.snapshot p{{margin-bottom:0}}.metrics{{grid-template-columns:repeat(2,1fr)}}.metric:nth-child(2){{border-right:0}}.metric:nth-child(-n+2){{border-bottom:1px solid var(--line)}}.two-col{{grid-template-columns:1fr;gap:0}}.flow{{grid-template-columns:repeat(2,1fr)}}.verdict-grid{{grid-template-columns:1fr;gap:4px}}nav{{gap:12px}}.panel{{padding:19px}}.splits{{gap:12px}}.split strong{{font-size:10px}}footer{{flex-direction:column;gap:8px}}}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
</style></head><body><main>
<header><div class="brand">FE CLUB <span>/ SimonResearch</span></div><nav aria-label="Dashboard sections"><a href="#performance">Performance</a><a href="#models">Models</a><a href="#method">Method</a></nav></header>
<section class="hero"><div><p class="eyebrow">A reproducible model showcase</p><h1>Nine features.<br><em>Signal, measured.</em></h1><p class="intro">Nonlinear stock forecasts, {selection_text}, and a trading simulation — built from the same daily data as the club notebook. Every result below comes from this recorded experiment.</p></div><aside class="snapshot" aria-label="Dataset coverage"><div><strong>{compact(dataset.get("tickers"))}</strong><p>STOCKS IN THE SNAPSHOT</p></div><div><strong>{compact(dataset.get("rows"))}</strong><p>MODEL-READY ROWS</p><span class="date-span">{date(dataset.get("start"))} – {date(dataset.get("end"))}</span></div></aside></section>
<div class="runline"><span class="status">● RECORDED EXPERIMENT</span><span>{escape(summary.get("experiment_id", "Unspecified experiment"))}</span><span>Generated {escape(summary.get("generated_at", "Timestamp unavailable"))}</span></div>
<section class="metrics" aria-label="Held-out ensemble metrics">{''.join(cards)}</section>
{verdict_panel(metrics, leaderboard)}
<section class="panel" id="performance"><div class="panel-head"><div><h2>Held-out portfolio performance</h2><p>Test-period equity against the equal-weight eligible-universe benchmark.</p></div><div class="segments" role="group" aria-label="Choose return treatment"><button type="button" data-equity="net" aria-pressed="true">After costs</button><button type="button" data-equity="gross" aria-pressed="false">Before costs</button></div></div><div id="equity-net" class="chart">{net}</div><div id="equity-gross" class="chart" hidden>{gross}</div><p class="chart-note">Equity uses the same common resolved holding dates for the strategy and benchmark. Signals observed at close enter at the next open and exit that session’s close; cash is held overnight. Planned allocations fund buy commissions; missing entry slots stay in cash. Recorded top-k: {top_k}; one-way commission assumption: {cost} bps; unresolved strategy dates: {unresolved}.</p>{conditional_note}</section>
<div class="two-col"><section class="panel"><div class="panel-head"><div><h2>Risk along the path</h2><p>A high return can still involve substantial losses.</p></div></div><div class="chart">{drawdown}</div></section><section class="panel"><div class="panel-head"><div><h2>Does the ranking carry signal?</h2><p>Rank IC measures alignment with realized stock-return ranks.</p></div></div><div class="chart">{ic}</div></section></div>
<section class="panel" id="models"><div class="panel-head"><div><h2>Held-out model comparison</h2><p>Simple baselines and nonlinear components share the same test window. A dash means a metric is unavailable or not applicable.</p></div></div>{board}<p class="chart-note">Forecast settings are selected on validation data; selection can favor one component rather than a blend. This test comparison describes the demo experiment; further iterations would make this window part of exploratory development.</p></section>
<div class="two-col"><section class="panel"><div class="panel-head"><div><h2>What the model uses</h2><p>Recorded model importance is descriptive and does not establish causality.</p></div></div><div class="chart">{importance}</div></section><section class="panel"><div class="panel-head"><div><h2>A prediction snapshot</h2><p>{prediction_date}. Illustrative model forecasts from the exported sample.</p></div></div>{prediction_table}</section></div>
<section class="panel" id="method"><div class="panel-head"><div><h2>One chronology. A traceable experiment.</h2><p>Stock observations stay together by date; validation controls model selection.</p></div></div><div class="splits">{''.join(split_cards)}</div><div class="flow"><div class="flow-item"><span>01 / INPUT</span><h3>Nine fixed features</h3><p>Returns, momentum, volatility, volume, range and intraday behavior, plus target-free same-day percentiles.</p></div><div class="flow-item"><span>02 / LEARNING</span><h3>Complementary forecasts</h3><p>XGBoost rank-target regression and LambdaMART ranking, CatBoost, and an Elastic Net anchor.</p></div><div class="flow-item"><span>03 / SELECTION</span><h3>Validation selection</h3><p>Candidate blends include single models. The recorded weights show the selected forecast.</p></div><div class="flow-item"><span>04 / EVALUATION</span><h3>Executable trade timing</h3><p>Forecast accuracy and trading returns are measured separately with recorded costs.</p></div></div><div class="weights" aria-label="Selected forecast weights">{weights_html}</div></section>
<section class="scope"><h2>Demo now, publication research next.</h2><p class="muted">This phase follows the user's choice to build a compelling model on the notebook data first. The later paper will require a confirmed literature contribution and a new research evaluation.</p><ul>{limitations_html}</ul></section>
<details><summary>Inspect experiment metadata and provenance</summary><p class="chart-note">Dataset SHA-256: {sha}</p><pre>{metadata}</pre></details>
<footer><span>FE Club / Fall 2026 · Personal research on SimonResearch</span><span>Self-contained report · Figures also saved as SVG and PNG</span></footer>
</main><script>
document.querySelectorAll('[data-equity]').forEach(button => {{button.addEventListener('click', () => {{const selected = button.dataset.equity;document.querySelectorAll('[data-equity]').forEach(item => item.setAttribute('aria-pressed', String(item.dataset.equity === selected)));document.getElementById('equity-net').hidden = selected !== 'net';document.getElementById('equity-gross').hidden = selected !== 'gross';}});}});
</script></body></html>'''
    path = report_dir / "index.html"
    path.write_text(output, encoding="utf-8")
    return path
