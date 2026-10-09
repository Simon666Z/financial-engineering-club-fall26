"""Guard the training boundary and target-free cross-sectional inference."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.preparation import FEATURES
from src.models.showcase import (
    COMPONENTS, FEATURE_NAMES, daily_rank_ic, make_features, make_rank_groups,
    fit_return_calibration, select_blend, train_showcase, transformed_feature_names,
)


def synthetic_frame(start, days=8, tickers=12, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    frame = pd.DataFrame({
        "Date": np.repeat(dates, tickers),
        "Ticker": [f"S{number:02}" for number in range(tickers)] * days,
    })
    for name in FEATURES:
        frame[name] = rng.normal(size=len(frame))
    frame["target"] = 0.025 * frame["ret_1d"] + rng.normal(scale=0.003, size=len(frame))
    frame["LabelEndDate"] = pd.to_datetime(frame["Date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(1, "D"))
    frame["target_rank"] = frame.groupby("Date")["target"].rank(pct=True)
    return frame


@pytest.fixture(scope="module")
def fitted_bundle():
    train = synthetic_frame("2022-09-01", days=30)
    validation = synthetic_frame("2023-03-01", days=10, seed=22)
    return train_showcase(train, validation, seed=7, threads=1, max_iterations=12)


def test_features_ignore_labels_sector_and_ticker_identity():
    frame = synthetic_frame("2024-01-02")
    first = make_features(frame)
    changed = frame.copy()
    changed["target"] = np.inf
    changed["target_rank"] = -999
    changed["Sector"] = "Future sector"
    changed["LabelEndDate"] = "2099-12-31"
    changed["Ticker"] = "renamed_" + changed["Ticker"]
    changed["Split"] = "arbitrary"
    pd.testing.assert_frame_equal(first, make_features(changed))
    assert first.columns.tolist() == FEATURE_NAMES
    assert len(first.columns) == 18


def test_daily_transforms_use_no_other_date_and_preserve_ties():
    frame = synthetic_frame("2024-01-02")
    first_date = frame["Date"].min()
    earlier = frame.loc[frame["Date"].eq(first_date)].copy()
    earlier.loc[:, "ret_1d"] = 2.0
    full = pd.concat([earlier, frame.loc[frame["Date"].gt(first_date)]], ignore_index=True)
    expected = make_features(earlier)
    observed = make_features(full).iloc[:len(earlier)].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, observed)
    assert (expected["ret_1d__csrank"] == 0).all()
    shuffled = full.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_features = make_features(shuffled)
    shuffled_features.index = pd.MultiIndex.from_frame(shuffled[["Date", "Ticker"]])
    ordered = make_features(full)
    ordered.index = pd.MultiIndex.from_frame(full[["Date", "Ticker"]])
    pd.testing.assert_frame_equal(shuffled_features.sort_index(), ordered.sort_index())


def test_rank_groups_are_sorted_and_do_not_break_target_ties():
    frame = synthetic_frame("2022-09-01", days=3, tickers=5)
    frame.loc[frame["Date"].eq(frame["Date"].min()), "target"] = 0.02
    shuffled = frame.sample(frac=1, random_state=3)
    ordered, labels, groups = make_rank_groups(shuffled)
    assert ordered[["Date", "Ticker"]].equals(
        ordered[["Date", "Ticker"]].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    )
    assert groups.tolist() == [5, 5, 5]
    assert len(set(labels[:5])) == 1
    assert labels.dtype.kind in "iu"
    assert labels.min() >= 0 and labels.max() <= 9


def test_validation_selects_single_model_when_ensemble_is_weaker():
    frame = synthetic_frame("2023-01-03", days=3, tickers=8)
    # All blends containing a reversed or flat component lose or tie; a single
    # exact ranking is available and wins the fixed simpler-candidate tie rule.
    good = frame.groupby("Date")["target"].rank(pct=True).to_numpy()
    bad = -good
    scores = {"elastic_net": good, "xgb_regression": bad, "xgb_ranker": bad, "catboost": np.zeros(len(frame))}
    weights, trials = select_blend(frame, scores)
    assert weights == {"elastic_net": 1.0}
    assert len(trials) == 12
    assert trials[0]["validation_rank_ic"]["mean"] == pytest.approx(1.0)
    flipped = frame.copy()
    flipped["target"] *= -1
    other_weights, _ = select_blend(flipped, scores)
    assert other_weights == {"xgb_regression": 1.0}


def test_rank_ic_counts_flat_or_too_small_days_as_undefined():
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-03"] * 3 + ["2023-01-04"] * 3 + ["2023-01-05"] * 2),
        "target": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.1, 0.2],
    })
    stats = daily_rank_ic(frame, [1, 2, 3, 1, 2, 3, 1, 2])
    assert stats == {"mean": 1.0, "scored_days": 1, "undefined_days": 2, "total_days": 3}


def test_test_labels_never_affect_predictions(fitted_bundle):
    bundle, metadata = fitted_bundle
    frame = synthetic_frame("2024-01-02", days=4)
    expected = bundle.predict_all(frame)
    changed = frame.drop(columns=["target", "target_rank", "LabelEndDate"])
    without_labels = bundle.predict_all(changed)
    pd.testing.assert_frame_equal(expected, without_labels)
    changed["target"] = np.nan
    changed["target_rank"] = -np.inf
    changed["LabelEndDate"] = "2099-12-31"
    pd.testing.assert_frame_equal(expected, bundle.predict_all(changed))
    assert metadata["test_labels_used"] is False
    assert metadata["fit_cutoff"] == "2022-12-31"
    assert metadata["selection_cutoff"] == "2023-12-31"
    assert metadata["refit_on_validation"] is False
    assert all(column in expected for column in COMPONENTS + ("ensemble", "ensemble_predicted_return"))
    assert expected[list(COMPONENTS) + ["ensemble"]].abs().max().max() <= 1
    assert bundle.return_calibration["slope"] >= 0


def test_future_dates_cannot_change_existing_predictions(fitted_bundle):
    bundle, _ = fitted_bundle
    first = synthetic_frame("2024-01-02", days=4)
    later = synthetic_frame("2025-01-02", days=4)
    expected = bundle.predict_all(first)
    observed = bundle.predict_all(pd.concat([first, later], ignore_index=True)).iloc[:len(first)]
    pd.testing.assert_frame_equal(expected, observed.reset_index(drop=True))


def test_saved_bundle_has_same_predictions(fitted_bundle, tmp_path):
    bundle, _ = fitted_bundle
    path = bundle.save(tmp_path / "showcase.joblib")
    loaded = joblib.load(path)
    frame = synthetic_frame("2024-01-02", days=3)
    pd.testing.assert_frame_equal(bundle.predict_all(frame), loaded.predict_all(frame))
    assert (tmp_path / "showcase_native" / "xgb_ranker.ubj").is_file()
    assert (tmp_path / "showcase_native" / "training_metadata.json").is_file()


@pytest.mark.parametrize("invalid", ["train_after_2022", "validation_after_2023", "crossing_label"])
def test_training_rejects_data_past_frozen_boundaries(invalid):
    train = synthetic_frame("2022-09-01")
    validation = synthetic_frame("2023-03-01")
    if invalid == "train_after_2022":
        train["Date"] += pd.DateOffset(years=1)
    elif invalid == "validation_after_2023":
        validation["Date"] += pd.DateOffset(years=1)
    else:
        train.loc[0, "LabelEndDate"] = "2023-01-03"
    with pytest.raises(ValueError):
        train_showcase(train, validation, max_iterations=2, threads=1)


def test_model_rejects_nonfinite_features_before_inference(fitted_bundle):
    bundle, _ = fitted_bundle
    frame = synthetic_frame("2024-01-02", days=3)
    frame.loc[0, "mom_20d"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        bundle.predict_all(frame)


def test_unlabeled_stocks_are_kept_in_training_and_validation_context():
    def add_unknown_stock(frame):
        unknown = frame.groupby("Date", sort=False).head(1).copy()
        unknown["Ticker"] = "UNKNOWN"
        unknown[FEATURES] = 100.0
        unknown["target"] = np.nan
        unknown["LabelEndDate"] = pd.to_datetime(np.full(len(unknown), np.datetime64("2099-01-01", "ns")))
        return pd.concat([frame, unknown], ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    train = add_unknown_stock(synthetic_frame("2022-09-01", days=12))
    validation = add_unknown_stock(synthetic_frame("2023-03-01", days=5, seed=22))
    bundle, metadata = train_showcase(train, validation, threads=1, max_iterations=5)
    known_train = train["target"].notna()
    expected = make_features(train).loc[known_train].mean().to_numpy()
    np.testing.assert_allclose(bundle.models["elastic_net"].named_steps["standardscaler"].mean_, expected, atol=1e-7)
    # The unlabeled high-valued stock changes every labeled stock's percentile,
    # so recomputing ranks after dropping it would violate this assertion.
    without_unknown = make_features(train.loc[known_train]).mean().to_numpy()
    assert not np.allclose(expected[len(FEATURES):], without_unknown[len(FEATURES):])
    known_validation = validation["target"].notna()
    scores = bundle.predict_all(validation).loc[known_validation].reset_index(drop=True)
    labeled = validation.loc[known_validation].reset_index(drop=True)
    for component in COMPONENTS:
        assert metadata["validation_rank_ic"][component] == daily_rank_ic(labeled, scores[component])
    assert bundle.return_calibration == fit_return_calibration(labeled, scores["ensemble"])
    assert metadata["unlabeled_context_rows"] == {"train": 12, "validation": 5}


def expanded_frame(start, days=8, seed=11):
    frame = synthetic_frame(start, days=days, seed=seed)
    frame["price_high_252d"] = frame["mom_60d"] / 10 + frame["ret_1d"] ** 2
    frame["volume_return_5d"] = frame["volume_ratio_20d"] * frame["ret_1d"]
    return frame


def test_expanded_feature_schema_is_explicit_ordered_and_target_free():
    features = [*FEATURES, "volume_return_5d", "price_high_252d"]
    frame = expanded_frame("2024-01-02")
    expected = make_features(frame, features=features)
    assert expected.columns.tolist() == transformed_feature_names(features)
    assert expected.columns.tolist()[:len(features)] == features
    changed = frame.drop(columns=["target", "target_rank", "LabelEndDate"])
    changed["unused_future_target"] = np.inf
    pd.testing.assert_frame_equal(expected, make_features(changed, features=features))
    # Merely adding columns never expands an existing nine-input schema.
    assert make_features(frame).columns.tolist() == FEATURE_NAMES


def test_expanded_bundle_round_trip_and_inference_without_test_labels(tmp_path):
    features = [*FEATURES, "price_high_252d", "volume_return_5d"]
    train = expanded_frame("2022-09-01", days=15)
    validation = expanded_frame("2023-03-01", days=5, seed=22)
    bundle, metadata = train_showcase(train, validation, threads=1, max_iterations=5, input_features=features)
    assert bundle.input_features == features
    assert bundle.feature_names == transformed_feature_names(features)
    assert metadata["input_features"] == features
    assert metadata["feature_names"] == bundle.feature_names
    assert metadata["added_features"] == ["price_high_252d", "volume_return_5d"]
    assert bundle.models["elastic_net"].named_steps["standardscaler"].n_features_in_ == 2 * len(features)
    expected_mean = make_features(train, features=features).mean().to_numpy()
    np.testing.assert_allclose(bundle.models["elastic_net"].named_steps["standardscaler"].mean_, expected_mean, atol=1e-7)
    frame = expanded_frame("2024-01-02", days=3)
    expected = bundle.predict_all(frame)
    pd.testing.assert_frame_equal(expected, bundle.predict_all(frame.drop(columns=["target", "target_rank", "LabelEndDate"])))
    loaded = joblib.load(bundle.save(tmp_path / "alpha.joblib"))
    assert loaded.input_features == features
    assert loaded.feature_names == bundle.feature_names
    pd.testing.assert_frame_equal(expected, loaded.predict_all(frame))
    with pytest.raises(ValueError, match="price_high_252d"):
        loaded.predict_all(frame.drop(columns=["price_high_252d"]))


def test_legacy_bundle_without_schema_keeps_original_predictions(fitted_bundle, tmp_path):
    bundle, _ = fitted_bundle
    original = bundle.predict_all(expanded_frame("2024-01-02", days=3))
    legacy = joblib.load(bundle.save(tmp_path / "legacy_source.joblib"))
    del legacy.input_features
    assert legacy.feature_names == FEATURE_NAMES
    pd.testing.assert_frame_equal(original, legacy.predict_all(expanded_frame("2024-01-02", days=3)))
    # Simulate an original saved object's state, which has no schema field.
    joblib.dump(legacy, tmp_path / "legacy_without_schema.joblib")
    restored = joblib.load(tmp_path / "legacy_without_schema.joblib")
    assert restored.input_features == FEATURES
    assert restored.feature_names == FEATURE_NAMES
    pd.testing.assert_frame_equal(original, restored.predict_all(expanded_frame("2024-01-02", days=3)))


@pytest.mark.parametrize("features", [
    [], ["ret_1d", "ret_1d"], ["target"], ["target_rank"], ["target_future"],
    ["Date"], ["Ticker"], ["LabelEndDate"], ["Split"], ["ret_1d__csrank"], "ret_1d",
])
def test_invalid_dynamic_schemas_are_rejected(features):
    with pytest.raises(ValueError):
        make_features(synthetic_frame("2024-01-02"), features=features)


def test_dynamic_training_can_use_a_schema_without_the_original_inputs():
    features = ["price_high_252d", "volume_return_5d"]
    required = ["Date", "Ticker", "target", "LabelEndDate", *features]
    train = expanded_frame("2022-09-01", days=12)[required]
    validation = expanded_frame("2023-03-01", days=4, seed=22)[required]
    bundle, _ = train_showcase(train, validation, threads=1, max_iterations=4, input_features=features)
    frame = expanded_frame("2024-01-02", days=2)[["Date", "Ticker", *features]]
    result = bundle.predict_all(frame)
    assert np.isfinite(result[list(COMPONENTS) + ["ensemble", "ensemble_predicted_return"]]).all().all()
    assert result[["reversal", "momentum"]].isna().all().all()


def test_expanded_features_keep_unlabeled_stocks_in_the_rank_population():
    features = [*FEATURES, "price_high_252d", "volume_return_5d"]

    def unknown_context(frame):
        unknown = frame.groupby("Date", sort=False).head(1).copy()
        unknown["Ticker"] = "UNKNOWN"
        unknown[features] = 100.0
        unknown["target"] = np.nan
        return pd.concat([frame, unknown], ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    train = unknown_context(expanded_frame("2022-09-01", days=10))
    validation = unknown_context(expanded_frame("2023-03-01", days=4, seed=22))
    bundle, metadata = train_showcase(train, validation, input_features=features, threads=1, max_iterations=4)
    known = train["target"].notna()
    expected_mean = make_features(train, features=features).loc[known].mean().to_numpy()
    np.testing.assert_allclose(bundle.models["elastic_net"].named_steps["standardscaler"].mean_, expected_mean, atol=1e-7)
    without_context = make_features(train.loc[known], features=features).mean().to_numpy()
    assert not np.allclose(expected_mean[len(features):], without_context[len(features):])
    known = validation["target"].notna()
    full_scores = bundle.predict_all(validation).loc[known].reset_index(drop=True)
    labeled = validation.loc[known].reset_index(drop=True)
    for name in COMPONENTS:
        assert metadata["validation_rank_ic"][name] == daily_rank_ic(labeled, full_scores[name])


def test_same_day_labels_are_rejected_before_model_fitting():
    train = synthetic_frame("2022-09-01")
    validation = synthetic_frame("2023-03-01")
    train["LabelEndDate"] = train["Date"]
    with pytest.raises(ValueError, match="not a later date"):
        train_showcase(train, validation, threads=1, max_iterations=2)
