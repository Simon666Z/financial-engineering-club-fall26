"""Canonical query padding never changes real date features or adds labels."""
import numpy as np
import pandas as pd
import pytest
from scripts.run_tabpfn import pad_query_features
from src.models.tabpfn_api import mutate_future_rows

def test_padding_preserves_real_values_order_schema_and_dtype():
    real=pd.DataFrame({"ret_1d":np.array([.04,-.02],dtype=np.float32),"ret_1d__csrank":np.array([1.,-1.],dtype=np.float32)},index=[10,11])
    result=pad_query_features(real,6)
    pd.testing.assert_frame_equal(result.iloc[:2].reset_index(drop=True),real.reset_index(drop=True))
    assert result.shape==(6,2) and (result.dtypes==np.dtype("float32")).all()
    assert np.count_nonzero(result.iloc[2:].to_numpy())==0
    assert not {"Date","Ticker","target"}.intersection(result.columns)

def test_tail_stress_preserves_anchor_and_alters_padding():
    real=pd.DataFrame(np.arange(8,dtype=np.float32).reshape(4,2),columns=["a","b"])
    query=pad_query_features(real,10)
    mutated=mutate_future_rows(query,2).astype(np.float32)
    pd.testing.assert_frame_equal(mutated.iloc[:2],query.iloc[:2])
    assert (mutated.iloc[4:].to_numpy()!=0).any()
    assert mutated.shape==query.shape and np.isfinite(mutated.to_numpy()).all()

def test_padding_rejects_truncation_and_nonfinite_values():
    real=pd.DataFrame({"a":[1.,2.]})
    with pytest.raises(ValueError): pad_query_features(real,1)
    with pytest.raises(ValueError): pad_query_features(real.assign(a=np.inf),3)


def test_resuming_preserves_preregistered_protocol_timestamp(tmp_path):
    import json
    from scripts.run_tabpfn import save_preflight
    first = {"generated_at": "2026-10-09T20:00:00Z", "input_fingerprint": "fixed",
             "model_config": {"fit_mode": "fit_with_cache"}, "query_protocol": {"query_rows": 10000}, "full_batches": [1]}
    save_preflight(tmp_path, first)
    original = (tmp_path / "preflight.json").read_bytes()
    latest = {**first, "generated_at": "2026-10-10T20:00:00Z", "usage_before": {"daily_tokens_used": 10000}}
    save_preflight(tmp_path, latest)
    assert (tmp_path / "preflight.json").read_bytes() == original
    assert json.loads((tmp_path / "preflight_latest.json").read_text()) == latest


def test_changed_preflight_cannot_overwrite_original_evidence(tmp_path):
    from scripts.run_tabpfn import save_preflight
    first = {"input_fingerprint": "fixed", "model_config": {}, "query_protocol": {}, "full_batches": [1]}
    save_preflight(tmp_path, first)
    original = (tmp_path / "preflight.json").read_bytes()
    with pytest.raises(ValueError, match="separate report directory"):
        save_preflight(tmp_path, {**first, "full_batches": [2]})
    assert (tmp_path / "preflight.json").read_bytes() == original
