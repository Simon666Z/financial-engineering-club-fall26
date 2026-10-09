"""Optional TabPFN-3.5 REST adapter; the existing local ensemble is untouched.

Wire schemas were checked against the official tabpfn-client 0.6.1 release.
The high-level SDK retries prediction timeouts. This adapter deliberately makes
one HTTP attempt per operation and sends no SDK telemetry. Standard inference
is used, without Thinking, automatic subsampling, or a KV-cache cost assumption.

A local token guard bounds quoted requests with 15% headroom. Prior Labs can
adjust a final charge after compute begins; its API has no documented hard
per-request token ceiling. A timeout therefore stops execution, and its charge
is treated as uncertain rather than retried.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

MODEL_VERSION = "v3.5"
MODEL_PATH = "v3.5_default"
N_ESTIMATORS = 4
SEED = 42
FEATURE_COUNT = 48
MAX_BATCH_ROWS = 10_000
PILOT_MAX_ROWS = 500
API_BASE = "https://api.priorlabs.ai"
SOURCES = [
    "https://docs.priorlabs.ai/models/selecting-model-version",
    "https://docs.priorlabs.ai/api/metering",
    "https://docs.priorlabs.ai/api/rate-limits",
    "https://docs.priorlabs.ai/api/security",
    "https://github.com/PriorLabs/tabpfn-client/blob/main/CHANGELOG.md",
    "https://pypi.org/project/tabpfn-client/0.6.1/",
    "https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v3_5.py",
    "https://arxiv.org/pdf/2605.13986",
    "https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf",
]
INPUT_FILES = (
    "train_features.parquet", "train_labels.npy", "validation_features.parquet",
    "test_features.parquet", "validation_keys.parquet", "test_keys.parquet",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_config() -> dict[str, Any]:
    return {"model_path": MODEL_PATH, "n_estimators": N_ESTIMATORS,
            "random_state": SEED, "fit_mode": "fit_preprocessors"}


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    names = manifest.get("feature_names")
    if not isinstance(names, list) or len(names) != FEATURE_COUNT or len(set(names)) != len(names):
        raise ValueError("Manifest must contain 48 distinct ordered feature_names.")
    forbidden = {"Date", "Ticker", "Sector", "Split", "target", "target_rank", "LabelEndDate"}
    if any(not isinstance(name, str) or not name or name in forbidden or name.startswith("target_") for name in names):
        raise ValueError("Feature schema includes an identifier or target field.")
    for name in ("train_end", "train_label_end"):
        date = pd.to_datetime(manifest.get(name), errors="coerce", utc=True)
        if pd.isna(date) or date > pd.Timestamp("2022-12-31 23:59:59", tz="UTC"):
            raise ValueError(f"Manifest {name} must be known and end in 2022 or earlier.")
    if manifest.get("transforms_before_sampling") is not True:
        raise ValueError("Same-date transformations must precede context sampling.")
    if manifest.get("seed") != SEED:
        raise ValueError("The fixed context seed must be 42.")
    if not isinstance(manifest.get("training_target_units"), str) or not manifest["training_target_units"]:
        raise ValueError("Manifest must document training_target_units.")
    return names


def validate_features(frame: pd.DataFrame, names: list[str], name: str) -> None:
    if list(frame.columns) != names or frame.columns.duplicated().any():
        raise ValueError(f"{name} does not match the ordered feature schema.")
    if frame.empty or any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"{name} must be a nonempty numeric feature table.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains missing or infinite inputs.")


def validate_keys(keys: pd.DataFrame, rows: int, split: str) -> pd.DataFrame:
    if list(keys.columns) != ["Date", "Ticker"] or len(keys) != rows or keys.empty:
        raise ValueError(f"{split} keys must be Date,Ticker with one key per feature row.")
    result = keys.copy().reset_index(drop=True)
    dates = pd.to_datetime(result["Date"], errors="coerce", utc=True)
    if dates.isna().any() or result["Ticker"].isna().any():
        raise ValueError(f"{split} keys are missing dates or tickers.")
    result["Date"] = dates.dt.tz_localize(None).dt.normalize()
    result["Ticker"] = result["Ticker"].astype(str)
    if result["Ticker"].str.len().eq(0).any() or result.duplicated(["Date", "Ticker"]).any():
        raise ValueError(f"{split} keys are blank or duplicated.")
    if not result.equals(result.sort_values(["Date", "Ticker"]).reset_index(drop=True)):
        raise ValueError(f"{split} keys must already be sorted by Date,Ticker.")
    low, high = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")) if split == "validation" else (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"))
    if result["Date"].lt(low).any() or result["Date"].gt(high).any():
        raise ValueError(f"{split} date boundary violated.")
    return result


@dataclass
class PreparedInputs:
    train: pd.DataFrame
    labels: np.ndarray
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_keys: pd.DataFrame
    test_keys: pd.DataFrame
    manifest: dict[str, Any]
    sha256: dict[str, str]

    @property
    def fingerprint(self) -> str:
        manifest = dict(self.manifest)
        manifest["sha256"] = {name: value for name, value in manifest.get("sha256", {}).items() if name in INPUT_FILES or name == "train_keys.parquet"}
        return canonical_hash({"manifest": manifest, "sha256": self.sha256, "model_config": model_config()})


def load_prepared(path: Path) -> PreparedInputs:
    manifest = json.loads((path / "manifest.json").read_text())
    names = validate_manifest(manifest)
    hashes = {name: sha256_file(path / name) for name in INPUT_FILES}
    expected = manifest.get("sha256", {})
    for name, actual in hashes.items():
        if name in expected and expected[name] != actual:
            raise ValueError(f"Prepared input checksum changed: {name}.")
    train, validation, test = [pd.read_parquet(path / f"{name}_features.parquet").reset_index(drop=True) for name in ("train", "validation", "test")]
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        validate_features(frame, names, name)
    labels = np.load(path / "train_labels.npy", allow_pickle=False)
    if labels.ndim != 1 or len(labels) != len(train) or not np.isfinite(labels).all():
        raise ValueError("Training labels must be finite, one-dimensional, and aligned.")
    if manifest.get("train_rows") != len(train):
        raise ValueError("Manifest train_rows disagrees with the training context.")
    train_keys_path = path / "train_keys.parquet"
    if train_keys_path.exists():
        hashes["train_keys.parquet"] = sha256_file(train_keys_path)
        if "train_keys.parquet" in expected and hashes["train_keys.parquet"] != expected["train_keys.parquet"]:
            raise ValueError("Training key checksum changed.")
        train_keys = pd.read_parquet(train_keys_path)
        if list(train_keys.columns) != ["Date", "Ticker"] or len(train_keys) != len(train):
            raise ValueError("Training context keys do not align with the context.")
        train_dates = pd.to_datetime(train_keys["Date"], errors="coerce", utc=True)
        if train_dates.isna().any() or train_dates.gt(pd.Timestamp("2022-12-31", tz="UTC")).any():
            raise ValueError("Training context includes future dates.")
        if train_keys.duplicated(["Date", "Ticker"]).any():
            raise ValueError("Training context contains duplicate keys.")
    validation_keys = validate_keys(pd.read_parquet(path / "validation_keys.parquet"), len(validation), "validation")
    test_keys = validate_keys(pd.read_parquet(path / "test_keys.parquet"), len(test), "test")
    return PreparedInputs(train, labels, validation, test, validation_keys, test_keys, manifest, hashes)


@dataclass(frozen=True)
class Batch:
    split: str
    start: int
    stop: int
    first_date: str
    last_date: str

    @property
    def rows(self) -> int:
        return self.stop - self.start

    def as_dict(self) -> dict[str, Any]:
        return {"split": self.split, "start": self.start, "stop": self.stop,
                "rows": self.rows, "first_date": self.first_date, "last_date": self.last_date}


def date_batches(keys: pd.DataFrame, split: str, max_rows: int = MAX_BATCH_ROWS) -> list[Batch]:
    if not 1 <= max_rows <= MAX_BATCH_ROWS:
        raise ValueError("Batch size must be between 1 and 10000.")
    groups = [(int(index.min()), int(index.max()) + 1) for index in keys.groupby("Date", sort=False).indices.values()]
    if not groups:
        raise ValueError("Cannot batch an empty key table.")
    result: list[Batch] = []
    start, stop = groups[0][0], groups[0][0]
    for begin, end in groups:
        if end - begin > max_rows:
            raise ValueError("A complete date cross-section exceeds the prediction batch limit.")
        if end - start > max_rows:
            result.append(Batch(split, start, stop, str(keys.iloc[start]["Date"].date()), str(keys.iloc[stop-1]["Date"].date())))
            start = begin
        stop = end
    result.append(Batch(split, start, stop, str(keys.iloc[start]["Date"].date()), str(keys.iloc[stop-1]["Date"].date())))
    return result


def pilot_batch(keys: pd.DataFrame) -> Batch:
    first = keys["Date"].iloc[0]
    rows = int(keys["Date"].eq(first).sum())
    if rows > PILOT_MAX_ROWS:
        raise ValueError("Pilot requires one complete validation date with at most 500 rows.")
    return Batch("pilot", 0, rows, str(first.date()), str(first.date()))


def causality_batches(keys: pd.DataFrame) -> list[Batch]:
    first = pilot_batch(keys)
    dates = keys["Date"].drop_duplicates().tolist()
    if len(dates) < 2:
        raise ValueError("Causality diagnostics require two whole validation dates.")
    end = int(keys["Date"].isin(dates[:2]).sum())
    if end > MAX_BATCH_ROWS:
        raise ValueError("Causality diagnostic query exceeds10000 rows.")
    last = str(dates[1].date())
    return [first, Batch("causality_combined", 0, end, first.first_date, last),
            Batch("causality_mutated", 0, end, first.first_date, last)]


def mutate_future_rows(features: pd.DataFrame, early_rows: int) -> pd.DataFrame:
    """Alter only later query rows; no target or training context is involved."""
    if not 0 < early_rows < len(features):
        raise ValueError("Causality mutation needs unchanged early and altered later rows.")
    # Promote explicitly before the stress mutation; every original float32
    # value remains exact and pandas does not emit per-column dtype warnings.
    changed = features.astype(np.float64)
    rng = np.random.default_rng(SEED)
    for column in changed.columns:
        values = changed[column].iloc[early_rows:].to_numpy(dtype=float)
        changed.loc[changed.index[early_rows:], column] = -11 * values[rng.permutation(len(values))] + 17
    if not np.isfinite(changed.to_numpy(dtype=float)).all():
        raise ValueError("Causality mutation produced nonfinite feature values.")
    return changed


def server_batch_limit(settings: dict[str, Any], train_rows: int, columns: int) -> int:
    limits = settings.get("model_limits", {}).get(MODEL_VERSION)
    if not isinstance(limits, dict):
        raise ValueError("Server did not advertise TabPFN-3.5 model limits.")
    checks = [(train_rows, "train_set_max_rows"), (columns, "max_cols"),
              (train_rows * columns, "train_set_max_cells"),
              (train_rows * columns, "train_set_max_upload_cells")]
    for actual, field in checks:
        limit = limits.get(field)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < actual:
            raise ValueError(f"Training context exceeds or lacks server limit {field}.")
    caps = [MAX_BATCH_ROWS]
    for field, divisor in [("test_set_max_rows", 1), ("test_set_max_cells", columns), ("predict_row_pairs_budget", train_rows)]:
        value = limits.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Missing positive server prediction limit {field}.")
        caps.append(value // divisor)
    if min(caps) < 1:
        raise ValueError("No test row fits the advertised server limits.")
    return min(caps)


def quote_payload(train_rows: int, test_rows: int, columns: int = FEATURE_COUNT) -> dict[str, Any]:
    if min(train_rows, test_rows, columns) < 1:
        raise ValueError("Quote dimensions must be positive.")
    return {"train_rows": train_rows, "test_rows": test_rows, "raw_columns": columns,
            "model_version": MODEL_VERSION, "operation": "predict", "n_estimators": N_ESTIMATORS}


def validate_quote(quote: dict[str, Any], expected: dict[str, Any]) -> int:
    cost = quote.get("estimated_cost")
    if quote.get("pricing_version") != "quota_v3" or type(cost) is not int or cost < 10_000:
        raise ValueError("A positive TabPFN-3.5 quota_v3 token quote is required.")
    if any(quote.get("inputs", {}).get(name) != value for name, value in expected.items()):
        raise ValueError("Provider quote resolved different dimensions or model settings.")
    return cost


@dataclass
class TokenBudget:
    maximum: int
    reserved: int = 0

    def __post_init__(self):
        if type(self.reserved) is not int or self.reserved < 0:
            raise ValueError("Previous token reservations must be nonnegative integers.")
        if type(self.maximum) is not int or self.maximum < 1:
            raise ValueError("An explicit positive --max-tokens cap is required.")

    @staticmethod
    def reservation(quoted_tokens: int) -> int:
        if type(quoted_tokens) is not int or quoted_tokens < 1:
            raise ValueError("Token estimates must be positive integers.")
        return math.ceil(quoted_tokens * 1.15)

    def check_plan(self, costs: list[int]) -> int:
        projected = sum(self.reservation(cost) for cost in costs)
        if self.reserved + projected > self.maximum:
            raise ValueError(f"Quoted requests plus 15% headroom need {self.reserved + projected:,} tokens; cap is {self.maximum:,}.")
        return projected

    def reserve(self, quoted_tokens: int) -> None:
        self.check_plan([quoted_tokens])
        # Keep the reservation on failure: computation may have started.
        self.reserved += self.reservation(quoted_tokens)


def validate_batch_independence(proof: dict[str, Any], fingerprint: str, record: dict[str, Any]) -> None:
    """Gate multi-date inference on documented architecture and deployment checks.

    Numerical checks corroborate the architecture; they cannot prove every
    possible future-query mutation. They must use this exact fit/configuration.
    """
    if proof.get("input_fingerprint") != fingerprint or proof.get("fitted_train_set_id") != record.get("fitted_train_set_id") or proof.get("model_config") != model_config():
        raise ValueError("Batch-independence evidence belongs to another context or model.")
    tolerance = proof.get("tolerance")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1e-6:
        raise ValueError("Batch-independence tolerance must be finite and at most1e-6.")
    if proof.get("status") != "passed" or proof.get("ranking_identical") is not True:
        raise ValueError("Batch-independence diagnostic must pass and preserve the early-date rankings.")
    for name in ("max_abs_future_mutation", "max_abs_date_alone"):
        difference = proof.get(name)
        if not isinstance(difference, (int, float)) or not math.isfinite(difference) or not 0 <= difference <= tolerance:
            raise ValueError(f"Batch-independence check failed or missing: {name}.")
    if type(proof.get("n_rows_checked")) is not int or proof["n_rows_checked"] < 2:
        raise ValueError("Batch-independence evidence needs a complete-date cross-section.")
    sources = proof.get("official_architecture_sources", [])
    required = {"https://arxiv.org/pdf/2605.13986", "https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf"}
    if not required.issubset(set(sources)):
        raise ValueError("Batch-independence evidence must cite both official architecture reports.")


def safe_usage(value: dict[str, Any]) -> dict[str, Any]:
    """Retain quota fields, excluding account identifiers and unrelated content."""
    result = {}
    for key, item in value.items():
        if any(word in key.lower() for word in ("usage", "limit", "remaining", "reset", "budget", "token", "quota", "pricing", "period", "daily", "monthly", "used", "current", "reserved", "charged", "spent")):
            if key.lower() in {"token", "refresh_token", "api_token"} or any(word in key.lower() for word in ("key", "secret", "access", "auth")):
                continue
            if isinstance(item, dict):
                result[key] = safe_usage(item)
            elif isinstance(item, (str, int, float, bool, type(None))):
                result[key] = item
    return result


class TabPFNAPI:
    """Single-attempt REST operations, with separate unauthenticated storage."""

    def __init__(self, api_key: str, *, timeout: float = 900, transport=None):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("TABPFN_API_KEY is required.")
        import httpx
        self._httpx = httpx
        self._api = httpx.Client(base_url=API_BASE, headers={"Authorization": f"Bearer {api_key}", "Prior-Client-Version": "0.6.1"}, timeout=timeout,
                                 follow_redirects=False, transport=transport or httpx.HTTPTransport(retries=0))
        self._storage = httpx.Client(timeout=timeout, follow_redirects=False,
                                    transport=transport or httpx.HTTPTransport(retries=0))
        self.last_headers: dict[str, str] = {}

    def close(self):
        self._api.close()
        self._storage.close()

    def _request(self, method: str, endpoint: str, **kwargs) -> tuple[dict[str, Any], int]:
        try:
            response = self._api.request(method, endpoint, **kwargs)
        except self._httpx.HTTPError as exc:
            raise RuntimeError(f"TabPFN {endpoint} transport failure ({type(exc).__name__}); no retry. A prediction charge may be uncertain.") from None
        self.last_headers = {key: response.headers[key] for key in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After", "X-Request-ID", "X-Trace-ID") if key in response.headers}
        if response.status_code not in (200, 201, 409):
            wait = response.headers.get("Retry-After")
            suffix = f" Retry-After: {wait}s." if wait and wait.isdigit() else ""
            raise RuntimeError(f"TabPFN {endpoint} returned HTTP {response.status_code}; no retry.{suffix}")
        try:
            value = response.json()
        except ValueError:
            raise RuntimeError(f"TabPFN {endpoint} returned invalid JSON; no retry.") from None
        if not isinstance(value, dict):
            raise RuntimeError(f"TabPFN {endpoint} returned an unexpected response.")
        return value, response.status_code

    def get_settings(self) -> dict[str, Any]:
        return self._request("GET", "/tabpfn/get_settings")[0]

    def get_usage(self) -> dict[str, Any]:
        return safe_usage(self._request("POST", "/get_api_usage/")[0])

    def estimate_cost(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/tabpfn/estimate_cost", json=payload)[0]

    @staticmethod
    def model_aliases(settings: dict[str, Any]) -> list[str]:
        # Like SDK list_available_models(), these names are aliases, not an
        # entitlement check or an immutable checkpoint identifier.
        return [f"{version}_default" for version in settings.get("model_limits", {})]

    def _storage_request(self, method: str, url: str, **kwargs):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not (parsed.hostname == "storage.googleapis.com" or (parsed.hostname or "").endswith(".storage.googleapis.com")):
            raise RuntimeError("Unexpected signed storage destination; upload/download stopped.")
        supplied = kwargs.get("headers", {})
        if any(name.lower() == "authorization" for name in supplied):
            raise RuntimeError("Storage requests must not contain Authorization headers.")
        try:
            response = self._storage.request(method, url, **kwargs)
        except self._httpx.HTTPError as exc:
            raise RuntimeError(f"TabPFN storage failure ({type(exc).__name__}); no retry.") from None
        if response.status_code not in (200, 201):
            raise RuntimeError(f"TabPFN storage returned HTTP {response.status_code}; no retry.")
        return response

    def _upload(self, content: bytes, info: dict[str, Any]) -> None:
        urls = info.get("signed_urls", [])
        if not urls:
            raise RuntimeError("Provider did not return signed upload URLs.")
        headers = info.get("required_headers", {})
        chunk_size = len(content) // len(urls)
        for index, url in enumerate(urls):
            start = index * chunk_size
            stop = start + chunk_size if index < len(urls)-1 else len(content)
            self._storage_request("PUT", url, content=content[start:stop], headers=headers)

    @staticmethod
    def _parquet(frame: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        frame.to_parquet(output, index=False, compression="zstd")
        return output.getvalue()

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
        x_content = self._parquet(features)
        y_content = self._parquet(pd.DataFrame({"target": labels}))
        prep, status = self._request("POST", "/tabpfn/prepare_train_set_upload", json={
            "x_train_info": {"format": "parquet", "size_bytes": len(x_content), "use_chunks": False},
            "y_train_info": {"format": "parquet", "size_bytes": len(y_content), "use_chunks": False},
            "description": "FE Club frozen training context through 2022; numerical next-day stock-rank regression"})
        if status != 409:
            self._upload(x_content, prep["x_train_info"])
            self._upload(y_content, prep["y_train_info"])
        response, _ = self._request("POST", "/tabpfn/fit", json={
            "train_set_upload_id": prep["train_set_upload_id"],
            "task_config": {"task": "regression", "tabpfn_config": model_config()},
            "tabpfn_systems": ["preprocessing", "text"]})
        record = {"fitted_train_set_id": response["fitted_train_set_id"], "model_config": model_config(), "fit_timings": response.get("timings"),
                  "fit_mode": "standard; Thinking disabled; no automatic subsampling"}
        if response.get("status") == "pending":
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                state, _ = self._request("POST", "/tabpfn/get_fit_status", json={"fitted_train_set_id": record["fitted_train_set_id"]})
                if state.get("status") == "completed":
                    record["fit_timings"] = state.get("timings")
                    break
                if state.get("status") != "pending":
                    raise RuntimeError("TabPFN fit failed; no automatic refit.")
                time.sleep(min(10, max(0.3, float(state.get("retry_in_secs") or 1))))
            else:
                raise RuntimeError("TabPFN fit status timed out; no automatic refit.")
        elif response.get("status") != "completed":
            raise RuntimeError("TabPFN fit did not complete; no automatic refit.")
        return record

    def predict(self, features: pd.DataFrame, record: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        content = self._parquet(features)
        prep, status = self._request("POST", "/tabpfn/prepare_test_set_upload", json={
            "fitted_train_set_id": record["fitted_train_set_id"],
            "x_test_info": {"format": "parquet", "size_bytes": len(content), "use_chunks": False}})
        if status != 409:
            self._upload(content, prep["x_test_info"])
        response, _ = self._request("POST", "/tabpfn/predict", json={
            "test_set_upload_id": prep["test_set_upload_id"], "fitted_train_set_id": record["fitted_train_set_id"],
            "task_config": {"task": "regression", "tabpfn_config": model_config(), "predict_params": {"output_type": "mean"}}})
        prediction = response.get("prediction")
        if prediction is None and "prediction_uri" in response:
            prediction = self._storage_request("GET", response["prediction_uri"]).json()
        try:
            values = np.asarray(prediction, dtype=float)
        except (ValueError, TypeError):
            raise RuntimeError("Unexpected TabPFN mean regression output.") from None
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or len(values) != len(features) or not np.isfinite(values).all():
            raise RuntimeError("TabPFN predictions are not finite, aligned mean regression scores.")
        metadata = response.get("metadata", {})
        if metadata.get("n_estimators", N_ESTIMATORS) != N_ESTIMATORS or metadata.get("billing_model_version", MODEL_VERSION) != MODEL_VERSION:
            raise RuntimeError("Provider used different model settings; stop before another charge.")
        if metadata.get("execution_mode") == "thinking":
            raise RuntimeError("Provider unexpectedly used Thinking mode; stop before another charge.")
        return values, {"provider_metadata": metadata, "timings": response.get("timings"), "usage_headers": dict(self.last_headers)}
