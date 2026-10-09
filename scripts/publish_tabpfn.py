#!/usr/bin/env python3
"""Publish a completed fixed-shape candidate for the showcase's strict loader."""
from pathlib import Path
import hashlib
import json
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "reports/tabpfn-padded"
    run = json.loads((source / "run.json").read_text())
    proof = json.loads((source / "batch_causality.json").read_text())
    if run.get("status") != "complete" or run.get("mode") != "full" or proof.get("status") != "passed":
        raise ValueError("Only a complete full run with a passed fixed-shape diagnostic can be published.")
    if run.get("query_protocol", {}).get("name") != "fixed_shape_train_only_padding_v1":
        raise ValueError("Unexpected query protocol.")
    names = ("batch_causality.json", "validation_predictions.parquet", "test_predictions.parquet")
    expected = {**run["output_sha256"], "batch_causality.json": run["batch_independence_proof_sha256"]}
    for name in names:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected[name]:
            raise ValueError("Source artifact checksum mismatch: " + name)
    destination = root / "reports/tabpfn"
    archive = destination / "archive/variable-shape-v1"
    existing = destination / "run.json"
    current = json.loads(existing.read_text()) if existing.exists() else {}
    if current.get("query_protocol") != run["query_protocol"] and not archive.exists():
        archive.mkdir(parents=True)
        for path in destination.iterdir():
            if path.name == "archive":
                continue
            target = archive / path.name
            shutil.copytree(path, target) if path.is_dir() else shutil.copy2(path, target)
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source / name, destination / name)
    run["source_report_dir"] = "reports/tabpfn-padded"
    (destination / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    status = {"status": "complete", "candidate_key": "tabpfn_3_5", "generated_at": run["generated_at"],
              "reason": "Full validation and test inference completed under the verified fixed-shape protocol.",
              "query_protocol": run["query_protocol"], "source_report_dir": run["source_report_dir"],
              "validation_rows": sum(item["batch"]["rows"] for item in run["requests"] if item["batch"]["split"] == "validation"),
              "test_rows": sum(item["batch"]["rows"] for item in run["requests"] if item["batch"]["split"] == "test"),
              "daily_tokens_used": run["usage_after"]["daily_tokens_used"]}
    (destination / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print("Published completed candidate. Run run_showcase.py --feature-set alpha --rebacktest for full provenance validation and trading results.")


if __name__ == "__main__":
    main()
