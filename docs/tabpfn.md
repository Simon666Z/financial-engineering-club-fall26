# TabPFN-3.5: completed research comparison

The full candidate completed on October 9, 2026: **122,255 validation forecasts across 249 dates** and **248,377 test forecasts across 502 dates**. Under the same long-only strategy as every other model, $1 million becomes **$3,525,086.84** in 2024–2025: **+252.51% cumulative return**, **1.392 annualized Sharpe**, and **-39.40% maximum drawdown**, excluding costs. The four-model blend remains unchanged. [All model results](../reports/alpha/RESULTS.md).

## Fixed comparison

| Setting | TabPFN candidate |
| --- | --- |
| Hosted selector | `v3.5_default`, resolved by the provider to TabPFN-3.5-Plus |
| Task and target | Regression mean of centered same-date next-day adjusted-close return rank, in `[-1,1]` |
| Features | Same 24 raw alpha features and 24 target-free same-day percentile transforms |
| Training context | 20,000 labeled rows, sampled approximately equally across training dates with seed 42 |
| Available labeled training rows | 712,313, used by the core local models |
| Training cutoff | Signal dates through 2022-12-29; labels available through 2022-12-30 |
| Validation | 2023-01-03 to 2023-12-28 |
| Test | 2024-01-02 to 2025-12-31; 501 realized return sessions |
| Model settings | `n_estimators=4`, `random_state=42`, `fit_mode="fit_with_cache"` |
| Query protocol | Exactly 10,000 × 48 float32 values per request, with constant zero padding at the tail |

Date and ticker identify rows; they are not predictors. Validation/test labels never enter the API adapter. Features and supervised training ranks are computed on complete eligible real date populations before selecting the training context. TabPFN outputs ranking scores, not calibrated decimal returns or probabilities. [Exact input features](alpha-features.md).

The 20,000-row context is smaller than the core models' training sample. This compares model and context size together; it does not isolate architecture. Context and model settings were frozen before full scoring. TabPFN remains a separate candidate outside the original validation-selected blend.

## Fixed query protocol and causality

The original variable-size protocol failed the predeclared `1e-6` / identical-ranking gate: combining two dates changed first-date scores by up to 0.0001440942 and changed 74 of 490 ranks. A separate cached fit with variable query sizes also failed, with maximum difference 0.0001571923. Those failed experiments remain recorded; their thresholds and results were not relabeled.

The successful protocol holds the entire query shape and dtype fixed. Real rows follow greedy chronological whole-date blocks ordered by `Date,Ticker`. Unused slots contain zeros; dummy outputs are discarded. Padding never enters training labels, eligible stock populations, or feature statistics. The [managed KV cache](https://docs.priorlabs.ai/capabilities/kv-cache) reuses training-side attention state for repeated prediction against the same fit.

A live implementation can assemble previously observed dates in the current block plus today's complete cross-section, then fill the unused tail with zeros. A new block starts if adding today's cross-section would exceed 10,000 rows. That decision uses only past/current row counts. The offline run replaces the constant tail with subsequent dates. Training-only preprocessing and test-to-training attention provide architectural support for independence from these later query values. [TabPFN-3 report](https://arxiv.org/pdf/2605.13986), [TabPFN-3.5 report](https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf).

Before full inference, three requests used the same fitted context and exactly 10,000 query rows: the first validation date plus zeros; the first whole validation block plus zeros; and the same matrix with every row after the first date strongly changed, including padding. The gate was unchanged:

| Check on the first date's 490 stocks | Measured result |
| --- | ---: |
| Maximum score difference: date alone versus full block | 0 |
| Maximum score difference after changing every later query row | 0 |
| Earlier-date rankings | Identical |
| Original tolerance | `1e-6` |
| Result | Passed |

This is observed invariance for the saved diagnostic and exact hosted fit. It is not a proof for every input, mutation or provider update. Hosted Plus has proprietary optimizations. A future fit/protocol change needs matching new evidence. No claim is made that padding alone explains the earlier failures.

## Setup and frozen inputs

Use a separate Python environment to preserve the local-model environment:

```bash
python3.12 -m venv .venv-tabpfn
.venv-tabpfn/bin/python -m pip install -r requirements-tabpfn.txt
```

The requirements pin `tabpfn-client==0.6.1`. Our adapter follows its REST schemas directly and imports no SDK inference or telemetry code. Supply `TABPFN_API_KEY` in the environment or ignored repository `.env`; `TABPFN_TOKEN` is also accepted. Environment values take priority. Keep credentials out of source, notebooks, reports and shell history.

Prepare the context from the frozen local alpha snapshot, without fetching market data:

```bash
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python scripts/prepare_tabpfn.py
```

The bundle in `data/processed/tabpfn/` contains ordered 48-column training/validation/test feature tables, training labels, aligned `Date,Ticker` key tables, and a manifest with source hashes, boundaries, sampling policy and target units. The adapter checks these identities. Only numeric features and training labels are uploaded; keys and validation/test labels stay local. [Provider security and retention](https://docs.priorlabs.ai/api/security).

## Reproduce

The completed outputs can be replayed **without new inference**:

```bash
.venv-tabpfn/bin/python scripts/publish_tabpfn.py
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python scripts/run_showcase.py --feature-set alpha --rebacktest
```

For a new inference run with the exact protocol, inspect quotes first, then request the diagnostic and full run explicitly:

```bash
.venv-tabpfn/bin/python scripts/run_tabpfn.py --use-cache --pad-query-rows 10000
.venv-tabpfn/bin/python scripts/run_tabpfn.py --use-cache --pad-query-rows 10000 --check-batch-causality --max-tokens 1000000
.venv-tabpfn/bin/python scripts/run_tabpfn.py --use-cache --pad-query-rows 10000 --run-full --max-tokens 1000000
.venv-tabpfn/bin/python scripts/publish_tabpfn.py
```

The default preflight sends dimensions/settings only and requests no inference. The first protocol record is preserved on resumption; fresh quota checks go into `preflight_latest.json`. A mismatched saved plan requires a separate report directory. Full inference requires the passed diagnostic to match the fit, inputs and protocol. Completed batches are reused only when their keys, hashes and provenance agree.

## Budget and completion evidence

The successful protocol made **3 diagnostic + 39 full prediction requests**, with every provider response reporting a cache hit. Full inference consumed **390,000 actual tokens**, and its diagnostic consumed **30,000**. Including the two earlier failed variable-shape diagnostics, total account usage was **480,000 tokens**. The fixed-protocol journal reserved 483,000 tokens, including 15% quote headroom; reservations across all three protocols total 552,000. No quota reset was needed.

The adapter quotes standard prediction cost to reserve for possible cache fallback. Provider tokens measure compute, and final charges can differ from quotes. The local cap is not a guaranteed hard provider billing ceiling. Failed or timed-out computation may consume tokens. [Metering](https://docs.priorlabs.ai/api/metering), [rate limits](https://docs.priorlabs.ai/api/rate-limits).

Every HTTP operation has one attempt. There is no automatic prediction retry or refit after a failure. The journal records reservations before submission and blocks uncertain requests until explicitly resolved. Each completed batch is saved immediately. The date-isolated fallback runner, `scripts/run_tabpfn_daily.py`, remains available if a future protocol fails its causal check; it has a separately quoted, larger request budget.

`models/tabpfn-padded/model.json` references the remote fit; its ignored request journal tracks the 42 requests. `reports/tabpfn-padded/` retains the original preflight, passed diagnostic, batch predictions/provider metadata, and completed aggregate outputs. `reports/tabpfn/` holds the published outputs with `source_report_dir="reports/tabpfn-padded"`. The failed original pilot is archived in `reports/tabpfn/archive/variable-shape-v1/`; the cached variable-shape failure remains in `reports/tabpfn-cached/`.

The showcase's strict consumer reconstructs the protocol fingerprint, uploaded query matrices and hashes; checks preflight timing, complete whole-date coverage, source/response/output hashes and exact aggregate equality; and rejects incomplete or mismatched candidates. An independent audit verified all 42 requests, the same resolved provider checkpoint, all 751 dates and removal of every dummy output. Credentials and signed storage URLs are excluded from artifacts. A remote fit reference is not a downloadable trained-weight checkpoint.

## Interpretation

TabPFN has the highest cumulative return in this comparison, but its 39.40% drawdown is larger and its Sharpe is lower than the blend's. Positive trading returns alone do not establish useful stock-selection alpha. The momentum baseline also outperforms the original blend in cumulative return. [Full comparison and trading assumptions](../reports/alpha/RESULTS.md).

The [September 2026 model release](https://priorlabs.ai/technical-reports/tabpfn-3-5) postdates the 2024–2025 test: this is a retrospective comparison, not a claim that this exact method was tradable then. The saved current-universe data retain survivorship/membership bias. The repeatedly inspected test window is exploratory; OHLC4 is simulated execution, and transaction costs are excluded. These limits remain attached to every result.
