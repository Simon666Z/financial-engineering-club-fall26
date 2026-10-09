"""Saved API candidates must never fabricate portfolio performance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.reporting.showcase import (
    EXTERNAL_CANDIDATE, external_candidate_inputs, load_external_candidate,
    model_name, model_results_table, canonical_digest, parquet_digest,
    PADDED_QUERY_PROTOCOL, padded_batch_plan, transaction_cost_comparison_note,
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExternalCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = self.root / "reports/tabpfn"
        self.report.mkdir(parents=True)
        self.manifest = {"sha256": "alpha-sha", "raw_snapshot_sha256": "raw-sha"}
        self.names = ["ret_1d"]
        self.keys = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-02"]), "Ticker": ["A", "B"]})

    def load(self):
        return load_external_candidate(self.root, manifest=self.manifest, feature_names=self.names, expected_keys=self.keys)

    def complete_fixture(self):
        for split in ["test", "validation"]:
            frame = self.keys.assign(prediction=[0.2, -0.1])
            frame.to_parquet(self.report / f"{split}_predictions.parquet", index=False)
        prepared = {}
        for name in ["alpha_model_data.parquet", "alpha_panel.parquet", "alpha_data_manifest.json"]:
            path = self.root / "data/processed" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            prepared[str(path.relative_to(self.root))] = digest(path)
        config = {"model_path": "v3.5_default", "n_estimators": 4, "random_state": 42, "fit_mode": "fit_preprocessors"}
        proof = {"status": "passed", "ranking_identical": True, "input_fingerprint": "context-id",
                 "fitted_train_set_id": "fit-id", "feature_names": self.names, "model_config": config,
                 "tolerance": 1e-6, "max_abs_date_alone": 0, "max_abs_future_mutation": 0}
        write_json(self.report / "batch_causality.json", proof)
        context = {"feature_names": self.names, "transforms_before_sampling": True, "train_rows": 20000,
                   "train_end": "2022-12-29", "train_label_end": "2022-12-30", "prepared_alpha_sha256": prepared,
                   "alpha_data_sha256": "alpha-sha", "raw_snapshot_sha256": "raw-sha"}
        metadata = {"status": "complete", "mode": "full", "candidate_key": EXTERNAL_CANDIDATE,
                    "alpha_data_sha256": "alpha-sha", "raw_snapshot_sha256": "raw-sha", "feature_names": self.names,
                    "context_rows": 20000, "training_context": context, "train_end": "2022-12-29", "train_label_end": "2022-12-30",
                    "model_path": "v3.5_default", "n_estimators": 4, "test_labels_used": False, "blend_changed": False,
                    "batch_independence_proof_sha256": digest(self.report / "batch_causality.json"), "batch_independence_proof": proof,
                    "input_fingerprint": "context-id", "fitted_train_set_id": "fit-id", "model_config": config,
                    "output_sha256": {f"{split}_predictions.parquet": digest(self.report / f"{split}_predictions.parquet") for split in ["test", "validation"]}}
        write_json(self.report / "run.json", metadata)
        return metadata

    def single_date_fixture(self):
        metadata = self.complete_fixture()
        self.validation_keys = pd.DataFrame({"Date": pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-04"]), "Ticker": ["A", "B", "A"]})
        self.keys = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]), "Ticker": ["A", "B", "A"]})
        prepared = self.root / "data/processed/tabpfn"
        prepared.mkdir(parents=True)
        train_keys = pd.DataFrame({"Date": pd.to_datetime(["2022-12-29"]), "Ticker": ["A"]})
        train_keys.to_parquet(prepared / "train_keys.parquet", index=False)
        train_keys.assign(ret_1d=0.1)[self.names].to_parquet(prepared / "train_features.parquet", index=False)
        (prepared / "train_labels.npy").write_bytes(b"frozen-training-label")
        for split, keys in [("validation", self.validation_keys), ("test", self.keys)]:
            keys.to_parquet(prepared / f"{split}_keys.parquet", index=False)
            pd.DataFrame({"ret_1d": [0.1, 0.2, 0.3]}).to_parquet(prepared / f"{split}_features.parquet", index=False)
            keys.assign(prediction=[0.2, -0.1, 0.4]).to_parquet(self.report / f"{split}_predictions.parquet", index=False)
        context = metadata["training_context"]
        context["train_rows"] = 1
        context["sha256"] = {path.name: digest(path) for path in prepared.iterdir()}
        write_json(prepared / "manifest.json", context)
        metadata["context_rows"] = 1
        fingerprint_manifest = dict(context)
        fingerprint_manifest["sha256"] = dict(context["sha256"])
        metadata["input_fingerprint"] = canonical_digest({"manifest": fingerprint_manifest, "sha256": context["sha256"], "model_config": metadata["model_config"]})
        metadata["batch_mode"] = "single_date"
        metadata["query_protocol"] = "single_date_query"
        metadata["source_report_dir"] = "reports/tabpfn-daily"
        metadata["model_config"] = {**metadata["model_config"], "fit_mode": "fit_with_cache"}
        metadata["run_fingerprint"] = canonical_digest({"input_fingerprint": metadata["input_fingerprint"], "model_config": metadata["model_config"], "query_protocol": "single_date_query"})
        metadata["requests"] = []
        metadata["previous_batch_independence_diagnostic"] = {"status": "failed"}
        # A single-date candidate does not falsely pass the failed diagnostic.
        failed = {**metadata["batch_independence_proof"], "status": "failed", "ranking_identical": False}
        write_json(self.report / "batch_causality.json", failed)
        for split, keys in [("validation", self.validation_keys), ("test", self.keys)]:
            features = pd.read_parquet(prepared / f"{split}_features.parquet")
            aggregate = pd.read_parquet(self.report / f"{split}_predictions.parquet")
            metadata["output_sha256"][f"{split}_predictions.parquet"] = digest(self.report / f"{split}_predictions.parquet")
            for day in keys["Date"].drop_duplicates():
                mask = keys["Date"].eq(day)
                indices = keys.index[mask]
                selected_keys = keys.loc[mask].reset_index(drop=True)
                selected_features = features.loc[mask].reset_index(drop=True)
                date = day.strftime("%Y-%m-%d")
                relative = f"batches/{split}-{date}.parquet"
                path = self.root / "reports/tabpfn-daily" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                aggregate.loc[mask].reset_index(drop=True).to_parquet(path, index=False)
                batch = {"split": split, "start": int(indices.min()), "stop": int(indices.max()) + 1,
                         "first_date": date, "last_date": date, "rows": len(selected_keys)}
                metadata["requests"].append({"split": split, "date": date, "first_date": date, "last_date": date,
                    "rows": len(selected_keys), "batch": batch, "status": "completed", "output_relative_path": relative,
                    "input_fingerprint": metadata["input_fingerprint"], "run_fingerprint": metadata["run_fingerprint"],
                    "fitted_train_set_id": metadata["fitted_train_set_id"], "model_config": metadata["model_config"],
                    "query_keys_sha256": parquet_digest(selected_keys), "query_features_sha256": parquet_digest(selected_features),
                    "predictions_sha256": digest(path)})
        write_json(self.report / "run.json", metadata)
        return metadata

    def padded_fixture(self):
        metadata = self.single_date_fixture()
        prepared = self.root / "data/processed/tabpfn"
        standard_fingerprint = metadata["input_fingerprint"]
        protocol = dict(PADDED_QUERY_PROTOCOL)
        metadata["batch_mode"] = "multi_date"
        metadata["query_protocol"] = protocol
        metadata["source_report_dir"] = "reports/tabpfn-padded"
        metadata["input_fingerprint"] = canonical_digest({"prepared_fingerprint": standard_fingerprint,
                                                         "model_config": metadata["model_config"], "query_protocol": protocol})
        metadata["requests"] = []
        proof = {"status": "passed", "ranking_identical": True, "input_fingerprint": metadata["input_fingerprint"],
                 "input_sha256": metadata["training_context"]["sha256"], "fitted_train_set_id": metadata["fitted_train_set_id"],
                 "feature_names": self.names, "model_config": metadata["model_config"], "query_protocol": protocol,
                 "query_rows_constant": 10000, "tolerance": 1e-6, "max_abs_date_alone": 0, "max_abs_future_mutation": 0}
        write_json(self.report / "batch_causality.json", proof)
        source = self.root / metadata["source_report_dir"]
        write_json(source / "batch_causality.json", proof)
        metadata["batch_independence_proof"] = proof
        metadata["batch_independence_proof_sha256"] = digest(self.report / "batch_causality.json")
        plan = padded_batch_plan(self.validation_keys, "validation") + padded_batch_plan(self.keys, "test")
        write_json(source / "preflight.json", {"generated_at": "2026-10-09T00:00:00+00:00", "query_protocol": protocol,
                                              "model_config": metadata["model_config"], "input_fingerprint": metadata["input_fingerprint"], "full_batches": plan})
        for batch in plan:
            split = batch["split"]
            features = pd.read_parquet(prepared / f"{split}_features.parquet").astype("float32")
            padding = pd.DataFrame({"ret_1d": [0.0] * (10000 - len(features))}).astype("float32")
            query = pd.concat([features, padding], ignore_index=True)
            uploaded = canonical_digest({"parquet_sha256": parquet_digest(query), "columns": self.names, "rows": 10000})
            stem = f"{split}-{batch['start']:06d}-{batch['stop']:06d}"
            prediction_path, response_path = source / "batches" / (stem + ".parquet"), source / "batches" / (stem + ".json")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            pd.read_parquet(self.report / f"{split}_predictions.parquet").to_parquet(prediction_path, index=False)
            response = {"batch": batch, "query_protocol": protocol, "input_fingerprint": metadata["input_fingerprint"],
                        "query_rows": 10000, "real_rows": batch["rows"], "uploaded_query_sha256": uploaded,
                        "provider_metadata": {"test_set_num_rows": 10000, "test_set_num_cols": len(self.names), "n_estimators": 4,
                                              "billing_model_version": "v3.5", "task": "regression"}}
            write_json(response_path, response)
            metadata["requests"].append({"batch": batch, "batch_id": canonical_digest(batch), "status": "completed",
                "fitted_train_set_id": metadata["fitted_train_set_id"], "query_protocol": protocol, "query_rows": 10000,
                "real_rows": batch["rows"], "uploaded_query_sha256": uploaded, "started_at": "2026-10-09T00:01:00+00:00",
                "predictions_sha256": digest(prediction_path), "response_metadata_sha256": digest(response_path)})
        for split in ["pilot", "causality_combined", "causality_mutated"]:
            metadata["requests"].append({"batch": {"split": split}, "status": "completed", "fitted_train_set_id": metadata["fitted_train_set_id"],
                                         "query_protocol": protocol, "query_rows": 10000})
        write_json(self.report / "run.json", metadata)
        return metadata

    def test_absent_candidate_is_optional(self):
        self.assertEqual(self.load(), ({}, None))

    def test_pilot_only_never_supplies_test_predictions(self):
        write_json(self.report / "status.json", {"status": "pilot_only", "context_rows": 20000, "reason": "Batch diagnostic failed."})
        details, predictions = self.load()
        self.assertEqual(details["status"], "pilot_only")
        self.assertFalse(details["performance_valid"])
        self.assertIsNone(predictions)

    def test_status_change_invalidates_input_fingerprint(self):
        write_json(self.report / "status.json", {"status": "pilot_only"})
        before = external_candidate_inputs(self.root)
        write_json(self.report / "status.json", {"status": "incomplete"})
        self.assertNotEqual(before, external_candidate_inputs(self.root))

    def test_completed_status_alone_is_incomplete(self):
        write_json(self.report / "status.json", {"status": "complete"})
        details, predictions = self.load()
        self.assertEqual(details["status"], "incomplete")
        self.assertIsNone(predictions)

    def test_verified_candidate_is_separate_from_core_forecasts(self):
        self.complete_fixture()
        details, predictions = self.load()
        self.assertEqual(details["status"], "complete")
        self.assertFalse(details["blend_inclusion"])
        self.assertEqual(list(predictions), ["Date", "Ticker", EXTERNAL_CANDIDATE])
        self.assertEqual(len(predictions), len(self.keys))

    def test_failed_proof_rejects_complete_candidate(self):
        metadata = self.complete_fixture()
        metadata["batch_independence_proof"]["status"] = "failed"
        write_json(self.report / "batch_causality.json", metadata["batch_independence_proof"])
        metadata["batch_independence_proof_sha256"] = digest(self.report / "batch_causality.json")
        write_json(self.report / "run.json", metadata)
        with self.assertRaisesRegex(ValueError, "diagnostic failed"):
            self.load()

    def test_schema_or_snapshot_mismatch_rejects_candidate(self):
        metadata = self.complete_fixture()
        for key, replacement in [("alpha_data_sha256", "wrong"), ("raw_snapshot_sha256", "wrong"), ("feature_names", ["wrong"]), ("blend_changed", True), ("test_labels_used", True)]:
            with self.subTest(key=key):
                changed = {**metadata, key: replacement}
                write_json(self.report / "run.json", changed)
                with self.assertRaises(ValueError):
                    self.load()

    def test_forecast_mutation_invalidates_checksum(self):
        self.complete_fixture()
        path = self.report / "test_predictions.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "forecast checksum"):
            self.load()

    def test_missing_or_duplicate_test_rows_reject_complete_status(self):
        metadata = self.complete_fixture()
        for forecast in [self.keys.iloc[:1].assign(prediction=0.2), pd.concat([self.keys, self.keys.iloc[:1]], ignore_index=True).assign(prediction=0.2)]:
            path = self.report / "test_predictions.parquet"
            forecast.to_parquet(path, index=False)
            metadata["output_sha256"][path.name] = digest(path)
            write_json(self.report / "run.json", metadata)
            with self.assertRaises(ValueError):
                self.load()

    def test_complete_single_date_run_retains_failed_multidate_history(self):
        self.single_date_fixture()
        details, predictions = self.load()
        self.assertEqual(details["status"], "complete")
        self.assertEqual(details["batch_mode"], "single_date")
        self.assertEqual(details["fit_mode"], "fit_with_cache")
        self.assertEqual(len(predictions), len(self.keys))
        self.assertEqual(json.loads((self.report / "batch_causality.json").read_text())["status"], "failed")

    def test_single_date_rejects_missing_duplicate_or_multidate_requests(self):
        metadata = self.single_date_fixture()
        for change in ["missing", "duplicate", "multidate", "future_features", "wrong_fit", "unsafe_path", "unknown_protocol"]:
            changed = json.loads(json.dumps(metadata))
            if change == "missing":
                changed["requests"].pop()
            elif change == "duplicate":
                changed["requests"].append(changed["requests"][0])
            elif change == "multidate":
                changed["requests"][0]["last_date"] = "2023-01-04"
            elif change == "future_features":
                changed["requests"][0]["query_features_sha256"] = "other-date"
            elif change == "wrong_fit":
                changed["requests"][0]["fitted_train_set_id"] = "other-fit"
            elif change == "unsafe_path":
                changed["requests"][0]["output_relative_path"] = "../../outside.parquet"
            else:
                changed["query_protocol"] = "unknown"
            write_json(self.report / "run.json", changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_single_date_batch_changes_invalidate_cache_fingerprint(self):
        metadata = self.single_date_fixture()
        before = external_candidate_inputs(self.root)
        path = self.root / metadata["source_report_dir"] / metadata["requests"][0]["output_relative_path"]
        path.write_bytes(path.read_bytes() + b"modified")
        self.assertNotEqual(before, external_candidate_inputs(self.root))
        with self.assertRaisesRegex(ValueError, "batch forecast checksum"):
            self.load()

    def test_waiting_quota_supersedes_prior_pilot_status(self):
        write_json(self.report / "status.json", {"status": "pilot_only", "reason": "Old pilot."})
        write_json(self.report / "run.json", {"mode": "full", "status": "waiting_quota", "batch_mode": "single_date",
                    "reason": "Daily quota reached after saved date requests.", "completed_requests": 400, "planned_requests": 751})
        details, predictions = self.load()
        self.assertEqual(details["status"], "waiting_quota")
        self.assertEqual(details["completed_requests"], 400)
        self.assertIsNone(predictions)
        self.assertFalse(details["performance_valid"])

    def test_new_cached_multidate_fit_requires_its_own_matching_passed_proof(self):
        metadata = self.complete_fixture()
        metadata["model_config"]["fit_mode"] = "fit_with_cache"
        metadata["batch_independence_proof"]["model_config"] = dict(metadata["model_config"])
        write_json(self.report / "batch_causality.json", metadata["batch_independence_proof"])
        metadata["batch_independence_proof_sha256"] = digest(self.report / "batch_causality.json")
        write_json(self.report / "run.json", metadata)
        details, _ = self.load()
        self.assertEqual(details["fit_mode"], "fit_with_cache")
        metadata["batch_independence_proof"]["model_config"]["fit_mode"] = "fit_preprocessors"
        write_json(self.report / "batch_causality.json", metadata["batch_independence_proof"])
        metadata["batch_independence_proof_sha256"] = digest(self.report / "batch_causality.json")
        write_json(self.report / "run.json", metadata)
        with self.assertRaisesRegex(ValueError, "different schema or model"):
            self.load()

    def test_complete_padded_run_matches_recorded_protocol_and_batches(self):
        self.padded_fixture()
        details, predictions = self.load()
        self.assertEqual(details["status"], "complete")
        self.assertEqual(details["query_protocol"], PADDED_QUERY_PROTOCOL)
        self.assertEqual(len(predictions), len(self.keys))
        self.assertFalse(details["blend_inclusion"])

    def test_padded_run_rejects_shape_matrix_coverage_and_fit_changes(self):
        metadata = self.padded_fixture()
        for change in ["query_shape", "matrix", "fit", "missing_batch", "duplicate_batch", "unknown_protocol", "before_registration", "partial_date"]:
            changed = json.loads(json.dumps(metadata))
            if change == "query_shape":
                changed["requests"][0]["query_rows"] = 9999
            elif change == "matrix":
                changed["requests"][0]["uploaded_query_sha256"] = "mutated-query"
            elif change == "fit":
                changed["requests"][0]["fitted_train_set_id"] = "another-fit"
            elif change == "missing_batch":
                changed["requests"].pop(0)
            elif change == "duplicate_batch":
                changed["requests"].append(changed["requests"][0])
            elif change == "unknown_protocol":
                changed["query_protocol"]["padding_value"] = 0.1
            elif change == "before_registration":
                changed["requests"][0]["started_at"] = "2026-10-08T23:00:00+00:00"
            else:
                changed["requests"][0]["batch"]["stop"] = 1
            write_json(self.report / "run.json", changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load()

    def test_padded_response_mutation_invalidates_cache_and_result(self):
        metadata = self.padded_fixture()
        before = external_candidate_inputs(self.root)
        batch = metadata["requests"][0]["batch"]
        path = self.root / metadata["source_report_dir"] / "batches" / f"{batch['split']}-{batch['start']:06d}-{batch['stop']:06d}.json"
        path.write_text(path.read_text() + " ")
        self.assertNotEqual(before, external_candidate_inputs(self.root))
        with self.assertRaisesRegex(ValueError, "response checksum"):
            self.load()

    def test_padded_registration_must_match_final_model_and_protocol(self):
        metadata = self.padded_fixture()
        path = self.root / metadata["source_report_dir"] / "preflight.json"
        preflight = json.loads(path.read_text())
        preflight["query_protocol"]["query_rows"] = 5000
        write_json(path, preflight)
        with self.assertRaisesRegex(ValueError, "preregistered"):
            self.load()

    def test_cost_return_column_is_adjacent_to_base_return(self):
        board = pd.DataFrame([{"model": "ensemble", "status": "resolved", "cumulative_net_return": 0.7,
            "sharpe_net": 1.8, "max_drawdown_net": -0.09, "final_net_equity": 1700000,
            "cumulative_return_after_costs": 0.65, "transaction_cost_bps": 10, "cost_adjusted_status": "resolved"}])
        rendered = model_results_table(board)
        expected_headers = ["Model", "Return before costs", "Return after costs", "Sharpe", "Max drawdown", "Ending NAV", "Status"]
        positions = [rendered.index(f'<th scope="col">{label}</th>') for label in expected_headers]
        self.assertEqual(positions, sorted(positions))
        for value in ["70.0%", "65.0%", "1.80", "-9.0%", "$1,700,000"]:
            self.assertIn(value, rendered)
        self.assertEqual(rendered.count("Return after costs"), 1)
        self.assertNotIn("final_equity_after_costs", rendered)

    def test_invalid_cost_replay_is_blank_and_reason_is_escaped(self):
        board = pd.DataFrame([{"model": "catboost", "status": "resolved", "cumulative_net_return": 0.71,
            "cumulative_return_after_costs": 0.99, "cost_adjusted_status": "invalid",
            "cost_adjusted_failure": 'Funding <breach> & unresolved quote.'}])
        rendered = model_results_table(board)
        self.assertIn("71.0%", rendered)
        self.assertNotIn("99.0%", rendered)
        self.assertIn("Return after costs unavailable: invalid", rendered)
        self.assertIn("Funding &lt;breach&gt; &amp; unresolved quote.", rendered)
        self.assertIn('aria-label="Return after costs unavailable:', rendered)
        self.assertNotIn("<breach>", rendered)

    def test_cost_return_is_independent_and_can_show_a_loss(self):
        board = pd.DataFrame([{"model": "reversal", "status": "invalid", "cumulative_net_return": 0.5,
            "cumulative_return_after_costs": -0.12, "cost_adjusted_status": "resolved"}])
        rendered = model_results_table(board)
        self.assertIn("-12.0%", rendered)
        self.assertNotIn("50.0%", rendered)
        self.assertEqual(rendered.count("<td>—</td>"), 4)

    def test_cost_note_separates_ten_bps_replay_from_zero_bps_base(self):
        board = pd.DataFrame([{"model": "ensemble", "cumulative_return_after_costs": 0.65, "transaction_cost_bps": 10}])
        summary = {"evaluation": {"cost_bps": 0}, "transaction_cost_comparison": {"cost_bps": 10}}
        note = transaction_cost_comparison_note(summary, board)
        self.assertIn("10 bps per buy and sell", note)
        self.assertIn("separate self-financing replay", note)
        self.assertIn("Return before costs, Sharpe, max drawdown, ending NAV and curves use 0 bps", note)
        original_board = board.drop(columns="cumulative_return_after_costs")
        self.assertEqual(transaction_cost_comparison_note(summary, original_board), "")
        original_table = model_results_table(original_board)
        self.assertNotIn("Return after costs", original_table)
        self.assertNotIn("Return before costs", original_table)
        self.assertIn('<th scope="col">Return</th>', original_table)

    def test_pilot_metrics_render_as_blanks_even_if_supplied(self):
        board = pd.DataFrame([{"model": EXTERNAL_CANDIDATE, "status": "pilot_only", "cumulative_net_return": 0.99,
                               "sharpe_net": 5.0, "max_drawdown_net": -0.01, "final_net_equity": 1990000}])
        rendered = model_results_table(board)
        self.assertIn("TabPFN-3.5", rendered)
        self.assertIn("Pilot only", rendered)
        self.assertEqual(rendered.count("<td>—</td>"), 4)
        self.assertNotIn("99.0%", rendered)
        self.assertEqual(model_name(EXTERNAL_CANDIDATE), "TabPFN-3.5")


if __name__ == "__main__":
    unittest.main()
