# TabPFN-3.5 research candidate

This optional experiment sends the saved numeric feature tables to Prior Labs for regression. It produces a separate candidate and leaves the frozen local ensemble and its model selection intact. Published benchmark results motivate trying the model; they do not establish an advantage for next-day stock ranking.

## Fixed comparison

| Setting | TabPFN candidate |
| --- | --- |
| Hosted selector | `v3.5_default`, which selects TabPFN-3.5-Plus |
| Task | Regression mean of the centered next-day stock-return rank |
| Inputs | The same 24 raw alpha features and 24 same-day percentile transforms as the local models |
| Training context | 20,000 labeled rows, sampled approximately equally across training dates with seed 42 |
| Available labeled training rows | 712,313, used by the core local models |
| Training cutoff | Signal dates and label availability through 2022 |
| Validation | 2023 |
| Test forecasts | 2024–2025, including eligible rows without a known future return |
| Estimators and seed | `n_estimators=4`, `random_state=42` |
| Inference | Standard, `fit_preprocessors`; Thinking and automatic context subsampling disabled |

Features are constructed over each complete observable eligible date population **before** choosing the training context. Training targets are next-day adjusted-close returns converted to centered same-date percentile ranks in `[-1,1]`. Date and ticker identify rows; they are not predictors. TabPFN's numeric output is a ranking score in this target's units, not a calibrated decimal return or a probability.

The 20,000-row context makes this a comparison of model and context size together. It is not a controlled test of architecture alone. Context selection is fixed before validation and test scoring; test labels never enter the adapter. The [alpha feature guide](alpha-features.md) describes the shared inputs.

## Setup and frozen inputs

Use a separate Python environment to preserve the existing local model environment:

```bash
python3.12 -m venv .venv-tabpfn
.venv-tabpfn/bin/python -m pip install -r requirements-tabpfn.txt
```

The requirements pin `tabpfn-client==0.6.1`. Our adapter follows that release's REST schemas directly, because its high-level prediction method automatically retries some timeouts. The adapter imports no SDK inference code or telemetry client.

Supply `TABPFN_API_KEY` through the process environment or the ignored repository `.env`. `TABPFN_TOKEN` is also accepted. Environment values take priority. Obtain a key through the [Prior Labs account page](https://platform.priorlabs.ai/account/api-keys); keep its value out of notebooks, commands saved in shell history, commits, and reports.

Prepare the fixed context from the saved alpha snapshot with the core environment; this performs no download or inference:

```bash
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python scripts/prepare_tabpfn.py
```

The runner requires the resulting frozen bundle in `data/processed/tabpfn/`:

```text
train_features.parquet       train_labels.npy
validation_features.parquet  validation_keys.parquet
test_features.parquet        test_keys.parquet
manifest.json
```

The optional `train_keys.parquet` permits an additional check of actual context dates. Each feature table contains the same 48 columns in the manifest's recorded order. Validation and test keys contain only `Date,Ticker`, sorted in that order and aligned row for row with features. The manifest records training and label cutoffs, context seed, sampling policy, target units, raw snapshot identity, feature provenance, and checksums. The runner checks these inputs and does not download or rebuild market data. Retain this bundle and its manifest to reproduce the same context; seed alone does not identify a context without its sampling procedure and source snapshot.

Market features and training labels are uploaded when fitting. Query features are uploaded for prediction. Row identifiers and validation/test labels are not uploaded. [Prior Labs security documentation](https://docs.priorlabs.ai/api/security) describes account isolation, encrypted transport/storage, and temporary retention for caching and reproducibility. The hosted fit record references a remote fit, so later reuse also depends on the provider retaining it.

## Run in stages

From the repository root, inspect model limits, current usage, and all planned quotes first:

```bash
.venv-tabpfn/bin/python scripts/run_tabpfn.py
```

This default preflight sends dataset dimensions and settings only. It uploads no feature values and requests no inference. It writes `reports/tabpfn/preflight.json` with server limits, model aliases, account usage, quotes, and the complete query plan. An advertised model alias does not itself prove account access or pin an immutable checkpoint. [Official model selection](https://docs.priorlabs.ai/models/selecting-model-version) documents the selector.

Run one complete first validation date, at most 500 stocks:

```bash
.venv-tabpfn/bin/python scripts/run_tabpfn.py --run-pilot --max-tokens 40000
```

Then test whether earlier-date predictions depend on later query rows:

```bash
.venv-tabpfn/bin/python scripts/run_tabpfn.py --check-batch-causality --max-tokens 40000
```

The diagnostic uses the same fitted context and three queries: the first date alone; the first two dates together; and the same combined shape with only the second date's features permuted and strongly changed. A completed pilot supplies the first query without paying for it again. Passing requires earlier-date predictions to differ by at most `1e-6` and their rankings to remain identical. The diagnostic records the exact fit, input fingerprint, feature schema, comparisons, tolerance, and primary sources in `reports/tabpfn/batch_causality.json`.

After that check passes, request the full candidate:

```bash
.venv-tabpfn/bin/python scripts/run_tabpfn.py --run-full --max-tokens 500000
```

The full run checks the saved diagnostic against the same fit and context, then predicts complete dates in batches of at most 10,000 rows. It remains blocked if matching passing evidence is absent. Whole-date batching preserves the observable feature cross-sections. All model/context choices remain fixed; reading validation and test forecasts does not change the local blend.

## Token accounting and interrupted runs

The explicit token cap covers cumulative adapter requests for the same frozen inputs, including the pilot and diagnostic. Before fitting or predicting, the runner quotes the plan and reserves **15% extra per prediction**. It obtains a fresh matching quote immediately before each charge. Any changed quote, mismatched model configuration, or insufficient local cap stops execution.

Prior Labs describes tokens as computation units rather than words or dollar amounts. Standard fits and uploads have no separate token charge; TabPFN-3.5 predictions have a minimum charge of 10,000 tokens. Account limits and current estimates come from the provider, not a local pricing formula. Quotes do not guarantee dataset eligibility. Failed or timed-out computation can still consume tokens. See [metering](https://docs.priorlabs.ai/api/metering) and [rate limits](https://docs.priorlabs.ai/api/rate-limits).

The cap bounds the quoted request plan with a reserve; the provider does not document a hard per-request token ceiling. Its final charge can be adjusted after computation begins, so this local cap cannot guarantee an absolute final billing ceiling. Actual usage is recorded before and after a completed run.

Every HTTP operation has one attempt. The adapter performs no automatic prediction retry or refit after a failure. It journals a reservation before prediction and retains it when computation is uncertain. Such a batch requires explicit resolution before another attempt. Successful batch forecasts are saved immediately and reused only after their keys, hashes, fit, and input provenance match. An interrupted full run remains incomplete. API credentials and signed storage URLs are excluded from saved artifacts; storage requests use a separate client without API Bearer authorization.

## Causality evidence and research limits

The public base-model architecture provides a reason to expect query-row independence. Its [v3.5 source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v3_5.py#L1114-L1121) excludes attention between test rows and forms keys and values from training rows. Its [numeric preprocessing](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v3_5.py#L2403-L2440) fits statistics using training rows. The [TabPFN-3 report](https://arxiv.org/pdf/2605.13986) describes training-only summaries and test-to-training attention; the [TabPFN-3.5 report](https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf) retains that overall design.

The hosted Plus pipeline includes proprietary optimizations. The three-query diagnostic corroborates the observed behavior of this deployment and fitted context; it is not a proof for every possible future-feature mutation or provider update. Small floating-point differences can occur between query shapes. If the diagnostic fails, full multi-date inference stops. Predicting one date per request avoids later dates within each call, but its request count and minimum-token charges require a new budget plan.

Prior Labs describes the default v3.5 foundation checkpoint's pretraining as synthetic. Its historical Real-TabPFN-2.5 variant used real-data fine-tuning, so the synthetic claim should not be generalized to every variant. Neither synthetic pretraining nor query independence removes leakage from our own feature, label, sampling, or trading pipeline.

TabPFN-3.5 was [released in September 2026](https://priorlabs.ai/technical-reports/tabpfn-3-5). A test on 2024–2025 is therefore a retrospective comparison using a subsequently available method. It is not a claim that this exact model could have been traded during those years. The saved current-universe data also retain survivorship limitations described in the research roadmap.

## Artifacts and completion

`models/tabpfn/model.json` stores the fitted remote reference and context provenance. `models/tabpfn/request_journal.json` records attempted requests and their reservations. `reports/tabpfn/batches/` stores completed batch forecasts and provider metadata.

A completed full run writes `validation_predictions.parquet` and `test_predictions.parquet`, each with `Date,Ticker,prediction`, plus `reports/tabpfn/run.json`. The candidate key is `tabpfn_3_5`. Only `status="complete"` identifies a fully generated candidate; metadata include data identities, fixed model settings, diagnostic evidence, output hashes, quotes, usage, and request identifiers when supplied. A remote fit reference is not a downloadable trained-weight checkpoint.

Measured execution status and comparison results are recorded separately after inference finishes. A successful API response or causal diagnostic alone does not establish useful investment performance.

## Measured pilot on October 9, 2026

The supplied account successfully fit the 20,000-row context and returned finite regression predictions for all 490 stocks on January 3, 2023. Three diagnostic prediction requests consumed **30,000 API tokens**; the cumulative local reservation was 34,500 tokens. No full-test predictions or TabPFN portfolio performance were generated.

The predeclared diagnostic **failed**:

| Check on January 3 predictions | Observed result |
| --- | ---: |
| Maximum score difference: date alone versus two-date query | 0.0001440942 |
| Changed ticker ranks | 74 / 490 |
| Largest rank movement | 2 places |
| Top20 overlap | 19 / 20 |
| Top100 overlap | 100 / 100 |
| Maximum score difference after same-shape future-feature mutation | 0 |

Changing query size swapped STLD and PSX across the rank20 entry boundary. Mutating the later-date features left earlier-date scores bit-identical. Query-shape numerical behavior is plausible, but these three requests do not isolate it from server nondeterminism or prove general independence. The original `1e-6` / identical-ranking gate was retained, so multi-date full inference stopped.

A conservative alternative sends one date per request. The saved inputs contain 249 validation dates and 502 test dates: 751 minimum-charge calls require at least **7.51 million quoted tokens**, exceeding the 500,000-token local experiment budget. That alternative requires a separately agreed budget and protocol. The optional candidate therefore appears as **pilot only**, with blank full-period performance, alongside the complete local-model comparison. The frozen four-model blend remains unchanged.

Measured evidence is saved in `reports/tabpfn/status.json`, `batch_causality.json`, `pilot_metadata.json`, and the local ignored batch prediction files. `reports/alpha/leaderboard.csv` remains the source of all completed trading results.
