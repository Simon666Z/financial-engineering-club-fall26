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
    model_name, model_results_table,
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
        config = {"model_path": "v3.5_default", "n_estimators": 4, "random_state": 42}
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
