"""Date-isolation, quota safety, and resumable API forecast contracts, all offline."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_tabpfn_daily as daily
from src.models.tabpfn_api import (
    MODEL_VERSION, TabPFNAPI, cached_model_config, model_config, quote_payload,
    validate_batch_independence, validate_model_config, validate_quote,
)


def test_cache_config_and_quote_preserve_frozen_default():
    before = model_config()
    cached = cached_model_config()
    assert before["fit_mode"] == "fit_preprocessors"
    assert cached == {**before, "fit_mode": "fit_with_cache"}
    validate_model_config(cached)
    assert model_config() == before
    payload = quote_payload(20_000, 490, operation="cache_predict")
    assert payload["operation"] == "cache_predict"
    assert validate_quote({"estimated_cost": 10_000, "pricing_version": "quota_v3", "inputs": payload}, payload) == 10_000
    with pytest.raises(ValueError):
        quote_payload(20_000, 490, operation="thinking_predict")
    with pytest.raises(ValueError):
        validate_model_config({**cached, "random_state": 7})


def test_cached_proof_requires_matching_config_and_unchanged_strict_gate():
    proof = {"status": "passed", "input_fingerprint": "frozen", "fitted_train_set_id": "cached",
             "model_config": cached_model_config(), "tolerance": 1e-6,
             "max_abs_future_mutation": 0, "max_abs_date_alone": 1e-8,
             "ranking_identical": True, "n_rows_checked": 490,
             "official_architecture_sources": ["https://arxiv.org/pdf/2605.13986", "https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf"]}
    record = {"fitted_train_set_id": "cached", "model_config": cached_model_config()}
    validate_batch_independence(proof, "frozen", record)
    for change in ({"model_config": model_config()}, {"ranking_identical": False}, {"max_abs_date_alone": 1e-5}):
        with pytest.raises(ValueError):
            validate_batch_independence({**proof, **change}, "frozen", record)


def test_date_isolated_queries_do_not_contain_future_dates():
    keys = pd.DataFrame({"Date": np.repeat(pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]), [2, 3, 1]),
                         "Ticker": ["A", "B", "A", "B", "C", "A"]})
    batches = daily.single_date_batches(keys, "validation")
    assert [(batch.start, batch.stop) for batch in batches] == [(0, 2), (2, 5), (5, 6)]
    for batch in batches:
        assert batch.first_date == batch.last_date
        assert keys.iloc[batch.start:batch.stop]["Date"].nunique() == 1
    broken = keys.iloc[[0, 2, 1, 3, 4, 5]].reset_index(drop=True)
    with pytest.raises(ValueError, match="contiguous"):
        daily.single_date_batches(broken, "validation")


def test_quota_guard_accounts_for_unreflected_charges_and_both_windows():
    usage = {"daily_tokens_used": 30_000, "daily_token_limit": 50_000,
             "monthly_tokens_used": 30_000, "monthly_token_limit": 200_000}
    assert daily.quota_block(usage, 11_500) is None
    assert daily.quota_block(usage, 11_500, 10_000) == "daily"
    assert daily.quota_block({**usage, "monthly_token_limit": 40_000}, 11_500) == "monthly"
    with pytest.raises(ValueError, match="monthly"):
        daily.quota_fields({"daily_tokens_used": 0, "daily_token_limit": 100})


def test_cache_fallback_reserves_more_expensive_standard_quote():
    class Quotes:
        def estimate_cost(self, payload):
            return {"estimated_cost": 10_000 if payload["operation"] == "cache_predict" else 12_000,
                    "pricing_version": "quota_v3", "inputs": payload}
    cost, quotes = daily.quoted_costs(Quotes(), 490, 48, 20_000)
    assert cost == 12_000
    assert quotes["cache_predict"]["estimated_cost"] == 10_000


@pytest.fixture
def offline_runner(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data/processed/tabpfn"
    data_dir.mkdir(parents=True)
    names = [f"feature_{index}" for index in range(48)]
    keys = pd.DataFrame({"Date": np.repeat(pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]), 3),
                         "Ticker": ["A", "B", "C"] * 3})
    for split, rows in (("train", 4), ("validation", 9), ("test", 9)):
        pd.DataFrame(np.arange(rows*48).reshape(rows, 48)/100, columns=names).to_parquet(data_dir/f"{split}_features.parquet", index=False)
    np.save(data_dir/"train_labels.npy", np.linspace(-1, 1, 4))
    keys.to_parquet(data_dir/"validation_keys.parquet", index=False)
    test_keys = keys.copy()
    test_keys["Date"] += pd.DateOffset(years=1)
    test_keys.to_parquet(data_dir/"test_keys.parquet", index=False)
    manifest = {"feature_names": names, "train_end": "2022-12-29", "train_label_end": "2022-12-30",
                "transforms_before_sampling": True, "seed": 42, "train_rows": 4,
                "training_target_units": "Centered next-day return rank [-1,1]"}
    (data_dir/"manifest.json").write_text(json.dumps(manifest))
    calls = []
    fits = []
    failure = []
    class FakeAPI:
        _parquet = staticmethod(TabPFNAPI._parquet)
        def __init__(self, key):
            pass
        def close(self):
            pass
        def get_settings(self):
            return {"dataset_max_size_bytes": 1_000_000,
                    "model_limits": {MODEL_VERSION: {"train_set_max_rows": 100_000, "train_set_max_cells": 10_000_000,
                    "train_set_max_upload_cells": 10_000_000, "max_cols": 48, "test_set_max_rows": 10_000,
                    "test_set_max_cells": 1_000_000, "predict_row_pairs_budget": 10_000_000}}}
        def get_usage(self):
            return {"daily_tokens_used": 10_000*len(calls), "daily_token_limit": 5_000_000,
                    "monthly_tokens_used": 10_000*len(calls), "monthly_token_limit": 20_000_000}
        def estimate_cost(self, payload):
            return {"estimated_cost": 10_000, "pricing_version": "quota_v3", "inputs": payload}
        def fit(self, features, labels, *, config):
            fits.append(config)
            return {"fitted_train_set_id": "fixed-cache", "model_config": config}
        def predict(self, features, record):
            calls.append(features.copy())
            if failure:
                raise RuntimeError("uncertain prediction; do not retry")
            return np.linspace(-0.5, 0.5, len(features)), {"provider_metadata": {"execution_mode": "cache", "cache_outcome": "hit"}, "usage_headers": {}, "timings": {}}
    monkeypatch.setattr(daily, "ROOT", root)
    monkeypatch.setattr(daily, "TabPFNAPI", FakeAPI)
    monkeypatch.setattr(daily, "load_api_key", lambda: "offline-test-placeholder")
    monkeypatch.setattr(daily.time, "sleep", lambda seconds: None)
    def invoke(*extras):
        monkeypatch.setattr(daily.sys, "argv", ["run_tabpfn_daily.py", "--data-dir", str(data_dir),
            "--model-dir", str(root/"models/tabpfn-daily"), "--report-dir", str(root/"reports/tabpfn-daily"),
            "--publish-dir", str(root/"reports/tabpfn"), *extras])
        daily.main()
    return SimpleNamespace(root=root, calls=calls, fits=fits, failure=failure, invoke=invoke)


def test_preflight_never_fits_or_predicts(offline_runner):
    offline_runner.invoke()
    assert not offline_runner.calls and not offline_runner.fits
    assert not (offline_runner.root/"reports/tabpfn/run.json").exists()


def test_smoke_resume_publishes_exact_full_coverage_without_repeating_calls(offline_runner):
    runner = offline_runner
    runner.invoke("--run", "--max-tokens", "100000", "--max-dates", "1")
    progress = json.loads((runner.root/"reports/tabpfn/run.json").read_text())
    assert progress["status"] == "incomplete" and progress["completed_requests"] == 1
    assert not (runner.root/"reports/tabpfn/test_predictions.parquet").exists()
    runner.invoke("--run", "--max-tokens", "100000")
    completed = json.loads((runner.root/"reports/tabpfn/run.json").read_text())
    assert completed["status"] == "complete"
    assert completed["batch_mode"] == "single_date"
    assert completed["query_protocol"] == "single_date_query"
    assert len(runner.calls) == 6 and len(runner.fits) == 1
    assert all(len(query) == 3 for query in runner.calls)
    assert completed["cumulative_reserved_tokens"] == 69_000
    for split in ("validation", "test"):
        assert len(pd.read_parquet(runner.root/f"reports/tabpfn/{split}_predictions.parquet")) == 9
    runner.invoke("--run", "--max-tokens", "100000")
    assert len(runner.calls) == 6 and len(runner.fits) == 1


def test_modified_saved_batch_stops_before_any_additional_charge(offline_runner):
    runner = offline_runner
    runner.invoke("--run", "--max-tokens", "100000", "--max-dates", "1")
    output = runner.root/"reports/tabpfn-daily/batches/validation-2023-01-03.parquet"
    frame = pd.read_parquet(output)
    frame.loc[0, "prediction"] += 0.1
    frame.to_parquet(output, index=False)
    with pytest.raises(ValueError, match="checksum"):
        runner.invoke("--run", "--max-tokens", "100000")
    assert len(runner.calls) == 1


def test_uncertain_prediction_is_journaled_and_never_retried(offline_runner):
    runner = offline_runner
    runner.failure.append(True)
    with pytest.raises(RuntimeError, match="uncertain"):
        runner.invoke("--run", "--max-tokens", "100000")
    run = json.loads((runner.root/"reports/tabpfn/run.json").read_text())
    assert run["status"] == "blocked_uncertain"
    assert run["cumulative_reserved_tokens"] == 11_500
    with pytest.raises(ValueError, match="uncertain"):
        runner.invoke("--run", "--max-tokens", "100000")
    assert len(runner.calls) == 1


def test_rest_fit_and_predict_bind_the_same_cached_configuration(monkeypatch):
    api = object.__new__(TabPFNAPI)
    calls = []
    def request(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs))
        if endpoint == "/tabpfn/prepare_train_set_upload":
            return {"train_set_upload_id": "train-upload", "x_train_info": {}, "y_train_info": {}}, 200
        if endpoint == "/tabpfn/fit":
            return {"fitted_train_set_id": "fixed-cache", "status": "completed"}, 200
        if endpoint == "/tabpfn/prepare_test_set_upload":
            return {"test_set_upload_id": "test-upload", "x_test_info": {}}, 200
        if endpoint == "/tabpfn/predict":
            return {"prediction": [0.1, 0.2], "metadata": {"n_estimators": 4, "billing_model_version": "v3.5",
                    "execution_mode": "cache", "cache_outcome": "hit", "tabpfn_config": {
                    **cached_model_config(), "model_path": "/app/models/resolved-v3.5.safetensors"}}}, 200
        raise AssertionError(endpoint)
    monkeypatch.setattr(api, "_request", request)
    monkeypatch.setattr(api, "_upload", lambda content, info: None)
    api.last_headers = {}
    features = pd.DataFrame({"A": [1.0, 2.0]})
    record = api.fit(features, np.array([0.1, 0.2]), config=cached_model_config())
    values, metadata = api.predict(features, record)
    assert np.array_equal(values, [0.1, 0.2])
    for _, endpoint, kwargs in calls:
        if endpoint in {"/tabpfn/fit", "/tabpfn/predict"}:
            assert kwargs["json"]["task_config"]["tabpfn_config"] == cached_model_config()
            assert "execution_mode" not in kwargs["json"]
            assert "cache_policy" not in kwargs["json"]
    assert metadata["provider_metadata"]["cache_outcome"] == "hit"
