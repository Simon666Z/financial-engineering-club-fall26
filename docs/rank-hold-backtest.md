# Persistent long/short backtest

This backtester applies the same persistent holding rules to any saved model forecast table. Execution-only runs reuse fitted models and predictions. The current [expanded-feature experiment](../reports/alpha/RESULTS.md) retrains the model separately, while preserving these trading rules and the original data snapshot.

## The daily sequence

1. Start with **$1,000,000 cash on 2 January 2024**. After that day's close, rank all eligible stocks using predictions based on information available then.
2. Plan long purchases in the **top 20** and short sales in the **bottom 20**. The first simulated fills occur on **3 January**, using that session's average price. Today's final features cannot support a trade using today's average.
3. Keep a long until its descending rank exceeds **100**. Keep a short until its ascending rank exceeds **100**. Rank 100 stays; rank 101 triggers an exit for the following session. The current setting also exits a holding that has no eligible rank.
4. Mark all remaining positions at the session close, calculate equity, and plan the next session's orders.

At each signal close, split **available free cash** 50/50 between new long and short allocations, equally within each selected set. Existing selected holdings can receive additions. Holdings still inside their exit threshold remain, so the portfolio can grow beyond 20 names per side. This is not a full daily rebalance: existing position sizes and long/short exposure can drift.

Only cash already available at the signal close funds its orders. Proceeds released by an exit on Wednesday become available at Wednesday's close for Thursday's orders. Wednesday's purchases cannot assume those sale proceeds beforehand. Missing entry fills leave the original allocation in cash without replacing the stock; missing exit fills retain the position for a later attempt.

## Cash and short positions

A long purchase spends cash. A short sale reserves an equal amount of entry cash as collateral, plus its commission; its sale proceeds are also reserved. Neither pool can finance extra purchases. Ignoring fees, the first $1 million funds $500,000 long exposure and $500,000 short exposure: approximately **100% total exposure**, not 200%.

For either the gross or net book:

```text
Equity = free cash + marked long assets
       + reserved short collateral + reserved short-sale proceeds
       - marked short liabilities - emergency funding debt
```

A falling shorted stock reduces its liability and increases equity. A rising one increases its liability and reduces equity. Gross and net results use independent cash and position ledgers, so fees change later allocations as well as current equity. Daily returns use the previous session's equity; capital never resets to $1 million.

## What the prices mean

The requested raw average is `OHLC4 = (Open + High + Low + Close) / 4`. It is known only after the execution day and is a **hypothetical fill**, not measured VWAP or a guaranteed attainable price.

To keep held inventory consistent across corporate actions, the simulation uses:

```text
AdjustmentFactor = Adj Close / Close
ExecutionAverage = AdjustmentFactor * OHLC4
MarkClose = Adj Close
```

The adjustment follows the price-scaling formula in [yfinance's auto_adjust implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/utils.py). Saved quantities represent synthetic adjusted total-return units, not verified historical share counts. No separate dividend cash or split adjustments are added, which would double-count adjustments already represented in these prices.

Open holdings are marked at the final observed close. There is no forced final sale, invented liquidation fee, or fill for the final signal beyond the snapshot.

## Assumptions and invalid results

Transaction costs are currently **disabled (0 basis points)** at Simon's request. The engine retains configurable fees for future runs; when enabled, each entry/exit is charged and entry budgets include the fee. The earlier 10 bps result is preserved in `reports/showcase/archive/hold-v2-10bps/`. Borrow fees default to **zero**; all selected stocks are assumed shortable. Real borrow availability, recalls, maintenance margin, spread and market impact are not modeled.

A missing held closing mark stops that portfolio's evaluation; the code does not substitute a stale price or zero return. A funding shortfall or insolvency invalidates headline performance statistics, blocks further new risk, and preserves the accounting ledger for inspection. A later accounting recovery does not restore valid performance.

The universe is the Wikipedia constituent snapshot, not historical S&P 500 membership. Yahoo prices can be revised. The 2024–2025 period was already viewed under earlier trading rules, so this revision is **exploratory**, not a fresh confirmation test for the future paper.

## Reproduce and inspect

From the research checkout, run:

```bash
.venv/bin/python scripts/run_showcase.py --feature-set alpha --rebacktest
```

This reuses frozen forecasts and recomputes execution. Settings are in [config/rank_hold.json](../config/rank_hold.json); `--retrain` and `--refresh-data` are separate explicit actions. The [research notebook](../notebooks/SimonResearch_Showcase.ipynb) presents the experiment. Inspect the current dashboard and audit outputs in `reports/alpha/`, including the summary, daily equity, trades and held positions. The first intraday experiment is preserved in `reports/showcase/archive/intraday-v1/`.
