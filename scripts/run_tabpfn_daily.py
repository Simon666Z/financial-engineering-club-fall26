#!/usr/bin/env python3
"""Predict every entire signal-date cross-section with a frozen TabPFN-3.5 context.

Default: dimensions/quota preflight only, no upload or inference. --run requires
an explicit cumulative --max-tokens cap. Cached training state improves latency;
each validation/test date remains a separate query. The earlier failed batch
independence diagnostic remains failed and is not used by this protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts.run_tabpfn import load_api_key, read_json
from src.models.tabpfn_api import (
    Batch, MODEL_PATH, MODEL_VERSION, N_ESTIMATORS, SOURCES, TabPFNAPI,
    TokenBudget, cached_model_config, canonical_hash, load_prepared,
    quote_payload, server_batch_limit, sha256_file, validate_quote, write_json,
)

PROTOCOL = "single_date_query"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frame_sha256(frame: pd.DataFrame) -> str:
    """Hash the exact deterministic feature/key parquet bytes, not target data."""
    return hashlib.sha256(TabPFNAPI._parquet(frame)).hexdigest()


def run_fingerprint(data) -> str:
    return canonical_hash({"input_fingerprint": data.fingerprint,
                           "model_config": cached_model_config(), "query_protocol": PROTOCOL})


def single_date_batches(keys: pd.DataFrame, split: str) -> list[Batch]:
    batches = []
    for date, offsets in keys.groupby("Date", sort=False).indices.items():
        start, stop = int(offsets.min()), int(offsets.max()) + 1
        if stop - start != len(offsets):
            raise ValueError("Every single-date query must be contiguous in the frozen key table.")
        day = str(pd.Timestamp(date).date())
        batches.append(Batch(split, start, stop, day, day))
    if not batches or sum(batch.rows for batch in batches) != len(keys):
        raise ValueError("Single-date plan does not cover every frozen key exactly once.")
    return batches


def quota_fields(usage: dict) -> dict:
    result = {}
    for period in ("daily", "monthly"):
        used = usage.get(f"{period}_tokens_used", usage.get(f"{period}_usage"))
        limit = usage.get(f"{period}_token_limit", usage.get(f"{period}_limit"))
        if type(used) is not int or type(limit) is not int or not 0 <= used <= limit:
            raise ValueError(f"Provider must advertise valid {period} quota usage and limit.")
        result[period] = {"used": used, "limit": limit, "remaining": limit - used,
                          "reset_at": usage.get(f"{period}_reset_time")}
    return result


def quota_block(usage: dict, required: int, unconfirmed: int = 0) -> str | None:
    fields = quota_fields(usage)
    # A quote reserve includes headroom. Predictions since the latest usage
    # snapshot are treated as spent at their worst quoted standard/cache cost.
    for period in ("monthly", "daily"):
        if fields[period]["remaining"] < required + unconfirmed:
            return period
    return None


def validate_cached_batch(item: dict, batch: Batch, output: Path, keys: pd.DataFrame,
                          features: pd.DataFrame, data, record: dict) -> pd.DataFrame:
    expected = {"input_fingerprint": data.fingerprint, "run_fingerprint": run_fingerprint(data),
                "model_config": cached_model_config(), "fitted_train_set_id": record["fitted_train_set_id"],
                "batch": batch.as_dict(), "query_features_sha256": frame_sha256(features),
                "query_keys_sha256": frame_sha256(keys)}
    if item.get("status") != "completed" or any(item.get(name) != value for name, value in expected.items()):
        raise ValueError("Saved prediction has incomplete or changed provenance; no automatic re-prediction.")
    if item.get("output_relative_path") != f"batches/{batch.split}-{batch.first_date}.parquet":
        raise ValueError("Saved prediction path differs from the single-date contract.")
    if not output.is_file() or item.get("predictions_sha256") != sha256_file(output):
        raise ValueError("Saved prediction checksum changed; no automatic re-prediction.")
    result = pd.read_parquet(output)
    if list(result.columns) != ["Date", "Ticker", "prediction"] or not result[["Date", "Ticker"]].equals(keys):
        raise ValueError("Saved prediction keys changed.")
    if not np.isfinite(result["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("Saved predictions must be finite.")
    return result


def quoted_costs(api, rows: int, columns: int, train_rows: int) -> tuple[int, dict]:
    quotes = {}
    costs = []
    for operation in ("cache_predict", "predict"):
        payload = quote_payload(train_rows, rows, columns, operation=operation)
        quote = api.estimate_cost(payload)
        costs.append(validate_quote(quote, payload))
        quotes[operation] = quote
    # The provider may use standard inference when a cache is unavailable.
    return max(costs), quotes


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly run/resume all frozen validation and test dates.")
    parser.add_argument("--max-tokens", type=int, help="Cumulative local quote cap including 15%% headroom and previous requests.")
    parser.add_argument("--max-dates", type=int, help="Smoke run: at most this many NEW signal dates, then checkpoint.")
    parser.add_argument("--wait-for-quota", action="store_true", help="Checkpoint and wait when the daily quota would be exceeded.")
    parser.add_argument("--min-request-interval", type=float, default=2.5, help="At least 2.5s between predictions, below1500/hour.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tabpfn")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/tabpfn-daily")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/tabpfn-daily")
    parser.add_argument("--publish-dir", type=Path, default=ROOT / "reports/tabpfn")
    args = parser.parse_args()
    if args.run and (args.max_tokens is None or args.max_tokens < 1):
        parser.error("--run requires a positive --max-tokens")
    if args.max_dates is not None and args.max_dates < 1:
        parser.error("--max-dates must be positive")
    if args.min_request_interval < 2.5:
        parser.error("--min-request-interval must be at least2.5 seconds")
    # Relative provenance is fixed and auditable by the dashboard consumer.
    source_report_dir = args.report_dir.resolve().relative_to(ROOT).as_posix()
    data = load_prepared(args.data_dir)
    fingerprint = run_fingerprint(data)
    batches = single_date_batches(data.validation_keys, "validation") + single_date_batches(data.test_keys, "test")
    api = TabPFNAPI(load_api_key())
    journal_path = args.model_dir / "request_journal.json"
    journal = read_json(journal_path, {"input_fingerprint": data.fingerprint, "run_fingerprint": fingerprint, "requests": []})
    if journal.get("input_fingerprint") != data.fingerprint or journal.get("run_fingerprint") != fingerprint:
        raise ValueError("The single-date journal belongs to another frozen context/configuration.")
    previous = {}
    for item in journal["requests"]:
        identifier = item.get("batch_id")
        if not identifier or identifier in previous:
            raise ValueError("The prediction journal has missing or duplicate request identifiers.")
        if item.get("status") != "completed":
            raise ValueError("A previous prediction has uncertain billing/output; resolve it explicitly before another attempt.")
        previous[identifier] = item
    planned = [(canonical_hash({"batch": batch.as_dict(), "run_fingerprint": fingerprint}), batch) for batch in batches]
    if set(previous) - {identifier for identifier, _ in planned}:
        raise ValueError("The prediction journal includes requests outside the declared date plan.")
    record = read_json(args.model_dir / "model.json", None)
    usage_before = None
    budget = None
    preflight = None

    def progress(status: str, **extra):
        value = {"status": status, "mode": "full", "candidate_key": "tabpfn_3_5", "batch_mode": "single_date",
                 "query_protocol": PROTOCOL, "generated_at": now(), "input_fingerprint": data.fingerprint,
                 "run_fingerprint": fingerprint, "model_config": cached_model_config(), "source_report_dir": source_report_dir,
                 "planned_requests": len(batches), "completed_requests": len(previous), "requests": journal["requests"],
                 "maximum_local_tokens": args.max_tokens, "cumulative_reserved_tokens": budget.reserved if budget else 0,
                 **extra}
        write_json(args.report_dir / "run.json", value)
        # A partial run always replaces the old pilot status in the dashboard.
        write_json(args.publish_dir / "run.json", value)

    try:
        settings = api.get_settings()
        usage_before = api.get_usage()
        limit = server_batch_limit(settings, len(data.train), data.train.shape[1])
        if max(batch.rows for batch in batches) > limit:
            raise ValueError("A full date cross-section exceeds the provider prediction limits.")
        byte_limit = settings.get("dataset_max_size_bytes")
        if type(byte_limit) is not int or byte_limit < max(int(data.train.memory_usage(deep=True).sum()), data.labels.nbytes):
            raise ValueError("Frozen training context exceeds the provider upload byte limit.")
        quotes = {}
        for rows in sorted({batch.rows for batch in batches}):
            cost, by_operation = quoted_costs(api, rows, data.train.shape[1], len(data.train))
            quotes[str(rows)] = {"maximum_cost": cost, "by_operation": by_operation}
        costs = [quotes[str(batch.rows)]["maximum_cost"] for _, batch in planned]
        pending = [(identifier, batch) for identifier, batch in planned if identifier not in previous]
        preflight = {"generated_at": now(), "mode": "metadata-only preflight", "input_fingerprint": data.fingerprint,
                     "run_fingerprint": fingerprint, "model_config": cached_model_config(), "batch_mode": "single_date",
                     "query_protocol": PROTOCOL, "training_rows": len(data.train), "feature_count": data.train.shape[1],
                     "planned_requests": len(batches), "remaining_requests": len(pending), "usage_before": usage_before,
                     "quota": quota_fields(usage_before), "quotes_by_test_rows": quotes,
                     "full_quoted_tokens": sum(costs), "full_reserved_tokens_with_headroom": sum(TokenBudget.reservation(cost) for cost in costs),
                     "cache_fallback_policy": "Reserve max(cache_predict,standard_predict); accept documented standard fallback within that reservation.",
                     "source_links": SOURCES + ["https://docs.priorlabs.ai/capabilities/kv-cache"]}
        write_json(args.report_dir / "preflight.json", preflight)
        print(f"Single-date TabPFN: {len(batches)} dates / {len(pending)} pending, {len(data.train):,} training rows / {data.train.shape[1]} inputs. {sum(costs):,} quoted tokens; {preflight['full_reserved_tokens_with_headroom']:,} including headroom.", flush=True)
        if not args.run:
            print("Preflight only: no fit, data upload, or inference.", flush=True)
            return
        budget = TokenBudget(args.max_tokens, reserved=sum(item["reserved_tokens"] for item in journal["requests"]))
        budget.check_plan([quotes[str(batch.rows)]["maximum_cost"] for _, batch in pending])
        if record is not None:
            if record.get("run_fingerprint") != fingerprint or record.get("input_fingerprint") != data.fingerprint or record.get("model_config") != cached_model_config():
                raise ValueError("Saved training cache belongs to another context/configuration; no automatic refit.")
        else:
            marker = args.model_dir / "fit_attempt.json"
            if marker.exists():
                raise ValueError("A previous fit lacks a saved completed record; no automatic refit.")
            write_json(marker, {"status": "started", "run_fingerprint": fingerprint, "started_at": now()})
            record = api.fit(data.train, data.labels, config=cached_model_config())
            record.update({"input_fingerprint": data.fingerprint, "run_fingerprint": fingerprint,
                           "input_sha256": data.sha256, "feature_names": data.manifest["feature_names"],
                           "training_context": data.manifest, "fitted_at": now(), "source_links": preflight["source_links"]})
            write_json(args.model_dir / "model.json", record)
            write_json(marker, {"status": "completed", "run_fingerprint": fingerprint, "completed_at": now()})
        # Verify all cached forecast bytes/provenance before making another call.
        accumulated = {"validation": [], "test": []}
        for identifier, batch in planned:
            if identifier not in previous:
                continue
            features = data.test if batch.split == "test" else data.validation
            keys = data.test_keys if batch.split == "test" else data.validation_keys
            selected = features.iloc[batch.start:batch.stop].reset_index(drop=True)
            selected_keys = keys.iloc[batch.start:batch.stop].reset_index(drop=True)
            output = args.report_dir / f"batches/{batch.split}-{batch.first_date}.parquet"
            validate_cached_batch(previous[identifier], batch, output, selected_keys, selected, data, record)
        progress("incomplete", fitted_train_set_id=record["fitted_train_set_id"])
        usage = api.get_usage()
        unconfirmed_cost = 0
        since_usage_check = 0
        last_started = 0.0
        executed = 0
        for identifier, batch in planned:
            if identifier in previous:
                continue
            if args.max_dates is not None and executed >= args.max_dates:
                break
            features = data.test if batch.split == "test" else data.validation
            keys = data.test_keys if batch.split == "test" else data.validation_keys
            selected = features.iloc[batch.start:batch.stop].reset_index(drop=True)
            selected_keys = keys.iloc[batch.start:batch.stop].reset_index(drop=True)
            if selected_keys["Date"].nunique() != 1 or len(selected_keys) != batch.rows:
                raise ValueError("A prediction query must contain exactly one WHOLE frozen signal date.")
            if int(selected.memory_usage(deep=True).sum()) > byte_limit:
                raise ValueError("Prediction cross-section exceeds upload byte limit.")
            fresh_cost, fresh_quotes = quoted_costs(api, batch.rows, data.train.shape[1], len(data.train))
            if fresh_cost != quotes[str(batch.rows)]["maximum_cost"]:
                raise ValueError("Provider cost changed after preflight; re-plan before charging.")
            reservation = TokenBudget.reservation(fresh_cost)
            blocked = quota_block(usage, reservation, unconfirmed_cost)
            if blocked:
                usage = api.get_usage()
                unconfirmed_cost = since_usage_check = 0
                blocked = quota_block(usage, reservation)
            while blocked:
                progress("waiting_quota", waiting_period=blocked, quota=quota_fields(usage), next_date=batch.first_date,
                         fitted_train_set_id=record["fitted_train_set_id"])
                print(f"Checkpoint: {len(previous)}/{len(batches)} dates complete. Waiting for {blocked} quota reset: {quota_fields(usage)[blocked]['reset_at']}.", flush=True)
                if blocked != "daily" or not args.wait_for_quota:
                    return
                # A short sleep allows interruption; no prediction is resubmitted.
                time.sleep(30)
                usage = api.get_usage()
                blocked = quota_block(usage, reservation)
            delay = args.min_request_interval - (time.monotonic() - last_started)
            if delay > 0:
                time.sleep(delay)
            budget.reserve(fresh_cost)
            relative = f"batches/{batch.split}-{batch.first_date}.parquet"
            output = args.report_dir / relative
            item = {"batch_id": identifier, "batch": batch.as_dict(), "split": batch.split, "date": batch.first_date,
                    "first_date": batch.first_date, "last_date": batch.last_date, "rows": batch.rows,
                    "input_fingerprint": data.fingerprint, "run_fingerprint": fingerprint, "model_config": cached_model_config(),
                    "fitted_train_set_id": record["fitted_train_set_id"], "query_features_sha256": frame_sha256(selected),
                    "query_keys_sha256": frame_sha256(selected_keys), "output_relative_path": relative,
                    "status": "submitted", "started_at": now(), "quoted_tokens": fresh_cost,
                    "reserved_tokens": reservation, "quote_by_operation": fresh_quotes}
            journal["requests"].append(item)
            write_json(journal_path, journal)
            last_started = time.monotonic()
            try:
                values, response_metadata = api.predict(selected, record)
                metadata = response_metadata["provider_metadata"]
                if metadata.get("execution_mode") not in {"cache", "standard"} or metadata.get("cache_outcome") not in {"hit", "miss", "fallback"}:
                    raise RuntimeError("Provider did not report a supported cache execution/fallback outcome; stop before another prediction.")
                prediction = selected_keys.copy()
                prediction["prediction"] = values
                write_prediction(output, prediction)
                metadata_path = output.with_suffix(".json")
                write_json(metadata_path, {**response_metadata, "batch": batch.as_dict(), "input_fingerprint": data.fingerprint,
                                          "run_fingerprint": fingerprint, "query_protocol": PROTOCOL,
                                          "query_features_sha256": item["query_features_sha256"], "query_keys_sha256": item["query_keys_sha256"]})
                item.update({"status": "completed", "completed_at": now(), "predictions_sha256": sha256_file(output),
                             "response_metadata_sha256": sha256_file(metadata_path), "execution_mode": metadata["execution_mode"],
                             "cache_outcome": metadata["cache_outcome"]})
                previous[identifier] = item
                write_json(journal_path, journal)
            except Exception:
                item.update({"status": "uncertain", "stopped_at": now()})
                write_json(journal_path, journal)
                progress("blocked_uncertain", fitted_train_set_id=record["fitted_train_set_id"])
                raise
            unconfirmed_cost += fresh_cost
            since_usage_check += 1
            executed += 1
            if since_usage_check >= 10:
                usage = api.get_usage()
                unconfirmed_cost = since_usage_check = 0
                progress("incomplete", fitted_train_set_id=record["fitted_train_set_id"], usage_after=usage)
            print(f"{len(previous)}/{len(batches)} {batch.split} {batch.first_date}: {batch.rows} stocks; {metadata['execution_mode']}/{metadata['cache_outcome']}.", flush=True)
        if len(previous) != len(planned):
            progress("incomplete", fitted_train_set_id=record["fitted_train_set_id"], usage_after=api.get_usage())
            print(f"Smoke checkpoint: {len(previous)}/{len(planned)} complete. Resume with the same --run/--max-tokens, without --max-dates.", flush=True)
            return
        # Publish only after every date is present, finite, and byte-verified.
        for identifier, batch in planned:
            features = data.test if batch.split == "test" else data.validation
            keys = data.test_keys if batch.split == "test" else data.validation_keys
            selected = features.iloc[batch.start:batch.stop].reset_index(drop=True)
            selected_keys = keys.iloc[batch.start:batch.stop].reset_index(drop=True)
            output = args.report_dir / f"batches/{batch.split}-{batch.first_date}.parquet"
            piece = validate_cached_batch(previous[identifier], batch, output, selected_keys, selected, data, record)
            accumulated[batch.split].append(piece)
        output_hashes = {}
        for split, pieces in accumulated.items():
            predictions = pd.concat(pieces, ignore_index=True)
            keys = data.test_keys if split == "test" else data.validation_keys
            if not predictions[["Date", "Ticker"]].equals(keys):
                raise ValueError("Complete predictions do not exactly cover the frozen chronological key table.")
            path = args.publish_dir / f"{split}_predictions.parquet"
            write_prediction(path, predictions)
            output_hashes[path.name] = sha256_file(path)
        old_diagnostic = read_json(ROOT / "reports/tabpfn/batch_causality.json", None)
        completed = {"generated_at": now(), "mode": "full", "status": "complete", "candidate_key": "tabpfn_3_5",
                     "batch_mode": "single_date", "query_protocol": PROTOCOL, "input_fingerprint": data.fingerprint,
                     "run_fingerprint": fingerprint, "model_config": cached_model_config(), "fitted_train_set_id": record["fitted_train_set_id"],
                     "feature_names": data.manifest["feature_names"], "training_context": data.manifest,
                     "input_sha256": data.sha256, "alpha_data_sha256": data.manifest.get("alpha_data_sha256"),
                     "raw_snapshot_sha256": data.manifest.get("raw_snapshot_sha256"), "train_rows": len(data.train),
                     "context_rows": len(data.train), "train_end": data.manifest["train_end"], "train_label_end": data.manifest["train_label_end"],
                     "model_path": MODEL_PATH, "model_version": MODEL_VERSION, "n_estimators": N_ESTIMATORS,
                     "prediction_units": "regression score; no predicted-return calibration", "output_sha256": output_hashes,
                     "source_report_dir": source_report_dir, "requests": journal["requests"], "planned_requests": len(batches),
                     "completed_requests": len(previous), "maximum_local_tokens": args.max_tokens, "cumulative_reserved_tokens": budget.reserved,
                     "usage_before": usage_before, "usage_after": api.get_usage(), "quotes_by_test_rows": quotes,
                     "previous_batch_independence_diagnostic": old_diagnostic, "source_links": preflight["source_links"],
                     "causal_note": "Each query contains one entire signal date only. Frozen training context ends in2022; no future query date or future target is supplied.",
                     "cache_fallback_policy": preflight["cache_fallback_policy"], "test_labels_used": False, "blend_changed": False,
                     "automatic_prediction_retries": 0, "historical_note": "TabPFN-3.5 released September2026; retrospective foundation-model comparison, unavailable during2024-25."}
        write_json(args.report_dir / "run.json", completed)
        write_json(args.publish_dir / "run.json", completed)
        print(f"COMPLETE: {len(previous)} single-date queries saved; ready for the unchanged long-only backtest.", flush=True)
    finally:
        api.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(1)
