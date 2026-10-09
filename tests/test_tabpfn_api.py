"""Offline checks for schema, chronological queries, budgets, and REST boundaries."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.tabpfn_api import (
    MODEL_VERSION, TabPFNAPI, TokenBudget, causality_batches, date_batches,
    load_prepared, model_config, mutate_future_rows, pilot_batch, quote_payload,
    safe_usage, server_batch_limit, sha256_file, validate_batch_independence,
    validate_features, validate_keys, validate_manifest, validate_quote,
)


@pytest.fixture
def names():
    return [f"feature_{index}" for index in range(48)]


@pytest.fixture
def manifest(names):
    return {"feature_names": names, "train_end": "2022-12-29", "train_label_end": "2022-12-30",
            "transforms_before_sampling": True, "seed": 42, "train_rows": 4,
            "training_target_units": "Centered next-day cross-sectional return rank [-1,1]"}


@pytest.fixture
def keys():
    return pd.DataFrame({"Date": np.repeat(pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]), 3),
                         "Ticker": ["A", "B", "C"] * 3})


@pytest.fixture
def prepared(tmp_path, names, manifest, keys):
    rng = np.random.default_rng(42)
    for split, rows in [("train", 4), ("validation", 9), ("test", 9)]:
        pd.DataFrame(rng.normal(size=(rows, 48)), columns=names).to_parquet(tmp_path / f"{split}_features.parquet", index=False)
    np.save(tmp_path / "train_labels.npy", np.linspace(-1, 1, 4))
    keys.to_parquet(tmp_path / "validation_keys.parquet", index=False)
    test_keys = keys.copy()
    test_keys["Date"] += pd.DateOffset(years=1)
    test_keys.to_parquet(tmp_path / "test_keys.parquet", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.fixture
def settings():
    return {"model_limits": {MODEL_VERSION: {"train_set_max_rows": 1_000_000, "train_set_max_cells": 100_000_000,
            "train_set_max_upload_cells": 100_000_000, "max_cols": 20_000, "test_set_max_rows": 1_000_000,
            "test_set_max_cells": 100_000_000, "predict_row_pairs_budget": 250_000_000_000}}}


def test_manifest_enforces_training_and_label_availability_cutoffs(manifest):
    assert len(validate_manifest(manifest)) == 48
    for field in ["train_end", "train_label_end"]:
        changed = dict(manifest, **{field: "2023-01-01"})
        with pytest.raises(ValueError, match=field):
            validate_manifest(changed)


@pytest.mark.parametrize("change", [{"transforms_before_sampling": False}, {"seed": 13}, {"training_target_units": ""}])
def test_manifest_requires_target_free_context_provenance(manifest, change):
    with pytest.raises(ValueError):
        validate_manifest(dict(manifest, **change))


def test_no_identifier_or_target_predictors(manifest):
    changed = copy.deepcopy(manifest)
    changed["feature_names"][0] = "target_rank"
    with pytest.raises(ValueError, match="target"):
        validate_manifest(changed)


def test_schema_order_and_finite_features_required(names):
    frame = pd.DataFrame(np.ones((3, 48)), columns=names)
    validate_features(frame, names, "train")
    with pytest.raises(ValueError, match="ordered"):
        validate_features(frame.iloc[:, ::-1], names, "train")
    frame.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_features(frame, names, "train")


@pytest.mark.parametrize("kind", ["future", "unsorted", "duplicate", "target_column"])
def test_query_keys_reject_wrong_dates_and_label_fields(keys, kind):
    changed = keys.copy()
    if kind == "future":
        changed.loc[0, "Date"] = pd.Timestamp("2024-01-03")
    elif kind == "unsorted":
        changed = changed.iloc[::-1]
    elif kind == "duplicate":
        changed.loc[1] = changed.iloc[0]
    else:
        changed["target"] = 1
    with pytest.raises(ValueError):
        validate_keys(changed, len(changed), "validation")


def test_complete_date_batches_preserve_every_row(keys):
    result = date_batches(keys, "validation", max_rows=7)
    assert [(batch.start, batch.stop) for batch in result] == [(0, 6), (6, 9)]
    assert [batch.rows for batch in result] == [6, 3]
    assert pilot_batch(keys).rows == 3
    with pytest.raises(ValueError, match="complete date"):
        date_batches(keys, "validation", max_rows=2)


def test_causality_mutation_preserves_first_date_and_batch_shape(names, keys):
    batches = causality_batches(keys)
    assert [batch.rows for batch in batches] == [3, 6, 6]
    frame = pd.DataFrame(np.arange(6 * 48).reshape(6, 48) / 100, columns=names)
    changed = mutate_future_rows(frame, 3)
    pd.testing.assert_frame_equal(changed.iloc[:3], frame.iloc[:3])
    assert changed.shape == frame.shape
    assert not np.array_equal(changed.iloc[3:], frame.iloc[3:])
    pd.testing.assert_frame_equal(changed, mutate_future_rows(frame, 3))


def test_frozen_input_hash_and_test_label_invariance(prepared):
    first = load_prepared(prepared)
    np.save(prepared / "test_targets.npy", [float("inf")])
    np.save(prepared / "validation_targets.npy", [float("nan")])
    changed_manifest = copy.deepcopy(first.manifest)
    changed_manifest["sha256"] = {"test_targets.parquet": "arbitrary", "validation_targets.parquet": "different"}
    (prepared / "manifest.json").write_text(json.dumps(changed_manifest))
    second = load_prepared(prepared)
    assert first.fingerprint == second.fingerprint
    pd.testing.assert_frame_equal(first.test, second.test)


def test_input_checksum_rejects_changed_features(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["sha256"] = {"train_features.parquet": sha256_file(prepared / "train_features.parquet")}
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    features = pd.read_parquet(prepared / "train_features.parquet")
    features.iloc[0, 0] += 1
    features.to_parquet(prepared / "train_features.parquet", index=False)
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(prepared)


def test_optional_actual_context_keys_reject_future_training_rows(prepared):
    pd.DataFrame({"Date": pd.to_datetime(["2022-12-01"] * 3 + ["2024-01-01"]), "Ticker": ["A", "B", "C", "D"]}).to_parquet(prepared / "train_keys.parquet", index=False)
    with pytest.raises(ValueError, match="future"):
        load_prepared(prepared)


def test_server_limits_include_cells_and_row_pair_work(settings):
    assert server_batch_limit(settings, 20_000, 48) == 10_000
    settings["model_limits"][MODEL_VERSION]["predict_row_pairs_budget"] = 20_000 * 200
    assert server_batch_limit(settings, 20_000, 48) == 200
    settings["model_limits"][MODEL_VERSION]["train_set_max_cells"] = 47 * 20_000
    with pytest.raises(ValueError, match="max_cells"):
        server_batch_limit(settings, 20_000, 48)


def test_quote_must_resolve_the_exact_fixed_model_and_dimensions():
    expected = quote_payload(20_000, 490)
    quote = {"estimated_cost": 10_000, "pricing_version": "quota_v3", "inputs": expected}
    assert validate_quote(quote, expected) == 10_000
    with pytest.raises(ValueError, match="quota_v3"):
        validate_quote(dict(quote, pricing_version="legacy_v2"), expected)
    with pytest.raises(ValueError, match="different"):
        validate_quote(dict(quote, inputs=dict(expected, n_estimators=8)), expected)


def test_cumulative_budget_includes_pilot_diagnostics_and_uncertain_calls():
    budget = TokenBudget(500_000)
    for _ in range(3):
        budget.reserve(10_000)
    assert budget.reserved == 34_500
    assert budget.check_plan([10_000] * 39) == 448_500
    assert budget.reserved + 448_500 == 483_000
    with pytest.raises(ValueError, match="cap"):
        budget.check_plan([10_000] * 41)
    # A failure does not return its reservation: the provider may have computed.
    budget.reserve(10_000)
    assert budget.reserved == 46_000


def test_batch_independence_requires_matching_fit_and_successful_checks():
    record = {"fitted_train_set_id": "model-one"}
    proof = {"status": "passed", "input_fingerprint": "input-one", "fitted_train_set_id": "model-one", "model_config": model_config(),
             "tolerance": 1e-6, "max_abs_future_mutation": 0.0, "max_abs_date_alone": 2e-8,
             "n_rows_checked": 490, "ranking_identical": True,
             "official_architecture_sources": ["https://arxiv.org/pdf/2605.13986", "https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf"]}
    validate_batch_independence(proof, "input-one", record)
    for changes in [{"fitted_train_set_id": "other"}, {"ranking_identical": False}, {"max_abs_future_mutation": 0.01}, {"tolerance": 0.1}, {"status": "failed"}]:
        with pytest.raises(ValueError):
            validate_batch_independence(dict(proof, **changes), "input-one", record)


def test_usage_metadata_excludes_credentials_and_account_identifiers():
    assert safe_usage({"email": "private", "access_token": "secret", "api_key": "secret", "current_usage": 3, "usage_limit": 5, "reset_time": "UTC"}) == {"current_usage": 3, "usage_limit": 5, "reset_time": "UTC"}


def fake_httpx(monkeypatch, *, raises=False):
    clients = []
    class HTTPError(Exception):
        pass
    class Client:
        def __init__(self, **kwargs):
            self.options, self.requests = kwargs, []
            clients.append(self)
        def request(self, method, endpoint, **kwargs):
            self.requests.append((method, endpoint, kwargs))
            if raises:
                raise HTTPError("sensitive-token-or-signed-url")
            return SimpleNamespace(status_code=200, headers={}, json=lambda: {"usage_limit": 10})
        def close(self):
            pass
    module = SimpleNamespace(Client=Client, HTTPError=HTTPError, HTTPTransport=lambda retries: {"retries": retries})
    monkeypatch.setitem(sys.modules, "httpx", module)
    return clients


def test_transport_has_zero_retries_and_does_not_repeat_uncertain_requests(monkeypatch):
    clients = fake_httpx(monkeypatch, raises=True)
    api = TabPFNAPI("test-only-dummy-key")
    assert clients[0].options["transport"] == {"retries": 0}
    with pytest.raises(RuntimeError, match="no retry") as caught:
        api.get_usage()
    assert len(clients[0].requests) == 1
    assert "sensitive-token" not in str(caught.value)
    assert "test-only-dummy-key" not in str(caught.value)
    api.close()


def test_signed_storage_never_receives_bearer_authorization(monkeypatch):
    clients = fake_httpx(monkeypatch)
    api = TabPFNAPI("test-only-dummy-key")
    api._storage_request("PUT", "https://storage.googleapis.com/example?signed=fake", content=b"example", headers={"Content-Type": "application/octet-stream"})
    assert "headers" not in clients[1].options
    assert "Authorization" not in clients[1].requests[0][2]["headers"]
    with pytest.raises(RuntimeError, match="Authorization"):
        api._storage_request("PUT", "https://storage.googleapis.com/example", headers={"Authorization": "anything"})
    with pytest.raises(RuntimeError, match="destination"):
        api._storage_request("PUT", "https://unrelated.example/example")
    api.close()
