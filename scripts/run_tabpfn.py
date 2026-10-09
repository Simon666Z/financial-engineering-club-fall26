#!/usr/bin/env python3
"""Preflight TabPFN-3.5 quotas; explicitly run a bounded chronological pilot/full candidate.

Use the isolated .venv-tabpfn interpreter. Default operation sends only
metadata: no training data upload or inference. --run-pilot and --run-full
require --max-tokens. The provider's quote is an estimate, so the local guard
reserves 15% extra per request; no API per-request hard ceiling is documented.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.models.tabpfn_api import (
    Batch, MODEL_PATH, MODEL_VERSION, N_ESTIMATORS, SOURCES, TabPFNAPI,
    TokenBudget, canonical_hash, causality_batches, date_batches, load_prepared, model_config, mutate_future_rows,
    pilot_batch, quote_payload, server_batch_limit, sha256_file, validate_quote,
    validate_batch_independence, write_json,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_api_key() -> str:
    # Never print or persist a token. Loading .env does not mutate the process
    # environment, and an explicitly supplied environment key takes priority.
    key = os.environ.get("TABPFN_API_KEY") or os.environ.get("TABPFN_TOKEN")
    if not key and (ROOT / ".env").is_file():
        from dotenv import dotenv_values
        values = dotenv_values(ROOT / ".env")
        key = values.get("TABPFN_API_KEY") or values.get("TABPFN_TOKEN")
    if not key:
        raise ValueError("Set TABPFN_API_KEY in the environment or the ignored repository .env.")
    return key


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-pilot", action="store_true", help="One complete first validation date, at most 500 rows.")
    mode.add_argument("--check-batch-causality", action="store_true", help="Three bounded queries to test earlier-row independence from later-date query features.")
    mode.add_argument("--run-full", action="store_true", help="Frozen context; complete 2023 validation and 2024-25 test, in whole-date batches.")
    parser.add_argument("--max-tokens", type=int, help="Explicit cumulative local quote budget, including 15%% headroom and previous adapter requests.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tabpfn")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/tabpfn")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/tabpfn")
    parser.add_argument("--batch-independence-proof", type=Path, help="Required for multi-date fullrun: architecture citations and checks for this fitted model.")
    args = parser.parse_args()
    if (args.run_pilot or args.run_full or args.check_batch_causality) and (args.max_tokens is None or args.max_tokens < 1):
        parser.error("--run-pilot/--run-full requires an explicit positive --max-tokens")

    if args.batch_independence_proof is None:
        args.batch_independence_proof = args.report_dir / "batch_causality.json"
    if args.run_full and not args.batch_independence_proof.is_file():
        parser.error("--run-full requires --batch-independence-proof; verify earlier predictions cannot depend on future query rows")
    data = load_prepared(args.data_dir)
    report, model_dir = args.report_dir, args.model_dir
    api = TabPFNAPI(load_api_key())
    try:
        settings = api.get_settings()
        usage_before = api.get_usage()
        limit = server_batch_limit(settings, len(data.train), data.train.shape[1])
        byte_limit = settings.get("dataset_max_size_bytes")
        if type(byte_limit) is not int or byte_limit < max(int(data.train.memory_usage(deep=True).sum()), data.labels.nbytes):
            raise ValueError("Training tables exceed or lack the server's upload byte limit.")
        pilot = pilot_batch(data.validation_keys)
        if pilot.rows > limit:
            raise ValueError("The complete pilot date exceeds server limits.")
        diagnostics = causality_batches(data.validation_keys)
        if max(batch.rows for batch in diagnostics) > limit:
            raise ValueError("Causality diagnostics exceed server prediction limits.")
        batches = date_batches(data.validation_keys, "validation", limit) + date_batches(data.test_keys, "test", limit)
        quotes = {}
        for rows in sorted({*(batch.rows for batch in diagnostics), *(batch.rows for batch in batches)}):
            payload = quote_payload(len(data.train), rows, data.train.shape[1])
            quote = api.estimate_cost(payload)
            validate_quote(quote, payload)
            quotes[str(rows)] = quote
        full_costs = [quotes[str(batch.rows)]["estimated_cost"] for batch in batches]
        preflight = {
            "generated_at": now(), "mode": "metadata-only preflight", "input_fingerprint": data.fingerprint,
            "model_selector": MODEL_PATH, "model_version": MODEL_VERSION, "model_config": model_config(),
            "available_aliases": api.model_aliases(settings), "alias_access_note": "Names do not guarantee account entitlement; selectors resolve provider checkpoints.",
            "settings": settings, "usage_before": usage_before, "quotes_by_test_rows": quotes,
            "pilot": pilot.as_dict(), "causality_batches": [batch.as_dict() for batch in diagnostics], "full_batches": [batch.as_dict() for batch in batches],
            "pilot_quoted_tokens": quotes[str(pilot.rows)]["estimated_cost"],
            "full_quoted_tokens": sum(full_costs), "full_reserved_tokens_with_headroom": sum(TokenBudget.reservation(cost) for cost in full_costs),
            "full_prediction_requests": len(batches), "diagnostic_quoted_tokens": sum(quotes[str(batch.rows)]["estimated_cost"] for batch in diagnostics), "source_links": SOURCES,
            "budget_note": "Local requests bounded by server quotes plus 15% headroom; provider final billing may differ. Timeout/failed compute is never automatically retried.",
            "historical_note": "TabPFN-3.5 was released September 2026; this is a retrospective foundation-model comparison, not a model available during 2024-25.",
        }
        write_json(report / "preflight.json", preflight)
        print(f"TabPFN-3.5 Plus, {len(data.train):,} context rows / {data.train.shape[1]} inputs / {N_ESTIMATORS} estimators.", flush=True)
        print(f"Pilot {pilot.rows} rows: {preflight['pilot_quoted_tokens']:,} quoted tokens. Full {len(batches)} prediction requests: {sum(full_costs):,} quoted / {preflight['full_reserved_tokens_with_headroom']:,} with headroom.", flush=True)
        if not (args.run_pilot or args.run_full or args.check_batch_causality):
            print("Preflight complete. No data upload or inference was requested.", flush=True)
            return

        chosen = diagnostics if args.check_batch_causality else ([pilot] if args.run_pilot else batches)
        journal_path = model_dir / "request_journal.json"
        journal = read_json(journal_path, {"input_fingerprint": data.fingerprint, "requests": []})
        if journal.get("input_fingerprint") != data.fingerprint:
            raise ValueError("Existing TabPFN request journal belongs to different data/configuration.")
        previous = {item["batch_id"]: item for item in journal["requests"]}
        budget = TokenBudget(args.max_tokens, reserved=sum(item["reserved_tokens"] for item in journal["requests"]))
        identifiers = {canonical_hash(batch.as_dict()): batch for batch in chosen}
        pending = [batch for identifier, batch in identifiers.items() if identifier not in previous]
        for identifier in identifiers:
            if identifier in previous and previous[identifier]["status"] != "completed":
                raise ValueError("A previous prediction has uncertain billing/output. Resolve it explicitly before another attempt; no retry.")
        budget.check_plan([quotes[str(batch.rows)]["estimated_cost"] for batch in pending])

        record_path = model_dir / "model.json"
        record = read_json(record_path, None)
        fit_marker = model_dir / "fit_attempt.json"
        if record is not None:
            if record.get("input_fingerprint") != data.fingerprint or record.get("model_config") != model_config():
                raise ValueError("Saved TabPFN fit belongs to different inputs or settings; no automatic refit.")
        else:
            if fit_marker.exists():
                raise ValueError("A prior fit has no saved completed record. Resolve it explicitly; no automatic refit.")
            write_json(fit_marker, {"input_fingerprint": data.fingerprint, "status": "started", "started_at": now()})
            record = api.fit(data.train, data.labels)
            record.update({"input_fingerprint": data.fingerprint, "input_sha256": data.sha256, "feature_names": data.manifest["feature_names"],
                           "training_context": data.manifest, "fitted_at": now(), "source_links": SOURCES})
            write_json(record_path, record)
            write_json(fit_marker, {"input_fingerprint": data.fingerprint, "status": "completed", "completed_at": now()})

        if args.run_full:
            validate_batch_independence(read_json(args.batch_independence_proof, None), data.fingerprint, record)

        accumulated: dict[str, list[pd.DataFrame]] = {}
        run_path = report / "run.json"
        if args.run_full:
            write_json(run_path, {"status": "incomplete", "started_at": now(), "candidate_key": "tabpfn_3_5", "input_fingerprint": data.fingerprint, "planned_requests": len(batches)})
        batch_dir = report / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for identifier, batch in identifiers.items():
            features = data.test if batch.split == "test" else data.validation
            keys = data.test_keys if batch.split == "test" else data.validation_keys
            selected = features.iloc[batch.start:batch.stop]
            if batch.split == "causality_mutated":
                selected = mutate_future_rows(selected, pilot.rows)
            selected_keys = keys.iloc[batch.start:batch.stop].reset_index(drop=True)
            output_path = batch_dir / f"{batch.split}-{batch.start:06d}-{batch.stop:06d}.parquet"
            metadata_path = output_path.with_suffix(".json")
            if int(selected.memory_usage(deep=True).sum()) > byte_limit:
                raise ValueError("Prediction batch exceeds upload byte cap.")
            if identifier in previous:
                item = previous[identifier]
                if item.get("fitted_train_set_id") != record["fitted_train_set_id"] or not output_path.is_file() or sha256_file(output_path) != item.get("predictions_sha256"):
                    raise ValueError("Completed batch provenance changed; no automatic re-prediction.")
                predictions = pd.read_parquet(output_path)
                if list(predictions.columns) != ["Date", "Ticker", "prediction"] or not predictions[["Date", "Ticker"]].equals(selected_keys) or not np.isfinite(predictions["prediction"]).all():
                    raise ValueError("Completed batch is not aligned with frozen keys.")
            else:
                # Re-quote with identical settings immediately before a charge.
                payload = quote_payload(len(data.train), batch.rows, data.train.shape[1])
                fresh = api.estimate_cost(payload)
                cost = validate_quote(fresh, payload)
                if cost != quotes[str(batch.rows)]["estimated_cost"]:
                    raise ValueError("Provider quote changed after preflight. Re-plan before charging.")
                budget.reserve(cost)
                item = {"batch_id": identifier, "batch": batch.as_dict(), "fitted_train_set_id": record["fitted_train_set_id"],
                        "status": "submitted", "started_at": now(), "quoted_tokens": cost, "reserved_tokens": TokenBudget.reservation(cost)}
                journal["requests"].append(item)
                write_json(journal_path, journal)
                try:
                    values, response_metadata = api.predict(selected, record)
                except Exception:
                    item.update({"status": "uncertain", "stopped_at": now()})
                    write_json(journal_path, journal)
                    raise
                predictions = selected_keys.copy()
                predictions["prediction"] = values
                predictions.to_parquet(output_path, index=False)
                response_metadata.update({"batch": batch.as_dict(), "input_fingerprint": data.fingerprint,
                                          "target_units": data.manifest["training_target_units"], "prediction_units": "regression mean of documented training target; ranking score, not calibrated decimal return"})
                write_json(metadata_path, response_metadata)
                item.update({"status": "completed", "completed_at": now(), "predictions_sha256": sha256_file(output_path),
                             "provider_request_ids": {name: value for name, value in response_metadata["usage_headers"].items() if name.lower() in {"x-request-id", "x-trace-id"}},
                             "response_metadata_sha256": sha256_file(metadata_path)})
                write_json(journal_path, journal)
            accumulated.setdefault(batch.split, []).append(predictions)
            print(f"{batch.split} {batch.first_date}..{batch.last_date}: {batch.rows} rows saved.", flush=True)

        if args.check_batch_causality:
            alone = accumulated["pilot"][0]["prediction"].to_numpy()
            combined = accumulated["causality_combined"][0]["prediction"].to_numpy()[:pilot.rows]
            altered = accumulated["causality_mutated"][0]["prediction"].to_numpy()[:pilot.rows]
            difference_alone = float(np.max(np.abs(combined - alone)))
            difference_mutation = float(np.max(np.abs(combined - altered)))
            ranks = [pd.Series(values).rank(method="average").to_numpy() for values in (alone, combined, altered)]
            ranking_identical = bool(np.array_equal(ranks[0], ranks[1]) and np.array_equal(ranks[1], ranks[2]))
            tolerance = 1e-6
            proof = {"status": "passed" if max(difference_alone, difference_mutation) <= tolerance and ranking_identical else "failed",
                     "input_fingerprint": data.fingerprint, "input_sha256": data.sha256,
                     "fitted_train_set_id": record["fitted_train_set_id"], "model_config": model_config(),
                     "model_version": MODEL_VERSION, "feature_names": data.manifest["feature_names"],
                     "first_date": pilot.first_date, "n_rows_checked": pilot.rows,
                     "max_abs_date_alone": difference_alone, "max_abs_future_mutation": difference_mutation,
                     "ranking_identical": ranking_identical, "tolerance": tolerance, "checked_at": now(),
                     "official_architecture_sources": ["https://arxiv.org/pdf/2605.13986", "https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf", "https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v3_5.py"],
                     "scope_note": "Base3.5 architecture has train-only statistics and test-to-train attention. These three queries corroborate this exact hostedPlus fit, without proving every possible query mutation."}
            write_json(args.batch_independence_proof, proof)
            validate_batch_independence(proof, data.fingerprint, record)
            print(f"Causality diagnostic PASS: same-shape future mutation maxdiff={difference_mutation:.3g}; alone maxdiff={difference_alone:.3g}; rankings identical.", flush=True)

        output_hashes = {}
        for split, pieces in accumulated.items():
            path = report / f"{split}_predictions.parquet"
            pd.concat(pieces, ignore_index=True).to_parquet(path, index=False)
            output_hashes[path.name] = sha256_file(path)
        completed = {"generated_at": now(), "mode": "causality_diagnostic" if args.check_batch_causality else ("pilot" if args.run_pilot else "full"), "status": "complete", "candidate_key": "tabpfn_3_5", "input_fingerprint": data.fingerprint,
                     "model_config": model_config(), "fitted_train_set_id": record["fitted_train_set_id"],
                     "feature_names": data.manifest["feature_names"], "training_context": data.manifest,
                     "alpha_data_sha256": data.manifest.get("alpha_data_sha256"), "raw_snapshot_sha256": data.manifest.get("raw_snapshot_sha256"),
                     "train_rows": len(data.train), "context_rows": len(data.train), "train_end": data.manifest["train_end"], "train_label_end": data.manifest["train_label_end"],
                     "model_path": MODEL_PATH, "n_estimators": N_ESTIMATORS,
                     "prediction_units": "regression score; no predicted-return calibration", "output_sha256": output_hashes,
                     "maximum_local_tokens": args.max_tokens, "cumulative_reserved_tokens": budget.reserved,
                     "batch_independence_proof": read_json(args.batch_independence_proof, None) if args.run_full or args.check_batch_causality else None,
                     "batch_independence_proof_sha256": sha256_file(args.batch_independence_proof) if args.run_full or args.check_batch_causality else None,
                     "requests": journal["requests"], "quotes_by_test_rows": quotes,
                     "usage_before": usage_before, "usage_after": api.get_usage(), "source_links": SOURCES,
                     "test_labels_used": False, "blend_changed": False, "automatic_prediction_retries": 0,
                     "budget_note": preflight["budget_note"], "historical_note": preflight["historical_note"]}
        metadata_name = "causality_metadata.json" if args.check_batch_causality else ("pilot_metadata.json" if args.run_pilot else "metadata.json")
        write_json(report / metadata_name, completed)
        if args.run_full:
            write_json(run_path, completed)
        print(f"Saved separate TabPFN candidate in {report}.", flush=True)
    finally:
        api.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        sys.exit(1)
