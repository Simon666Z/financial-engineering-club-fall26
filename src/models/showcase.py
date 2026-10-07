"""Small, fixed-budget stock-ranking ensemble for the daily starter data.

The models only use the notebook's nine features and each feature's observable
same-day percentile. Training ends in 2022. The 2023 validation set chooses the
best iteration, blend, and return calibration. No test labels enter this module.

Scores rank stocks; they are not returns or probabilities. A separate affine
calibration maps the ensemble score to a decimal next-day return estimate.
"""

from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker, XGBRegressor

from src.data.preparation import FEATURES


COMPONENTS = ("elastic_net", "xgb_regression", "xgb_ranker", "catboost")
FEATURE_NAMES = list(FEATURES) + [f"{name}__csrank" for name in FEATURES]
TRAIN_END = pd.Timestamp("2022-12-31")
VALIDATION_END = pd.Timestamp("2023-12-31")


def _dates(values):
    dates = pd.to_datetime(values, errors="coerce", utc=True)
    return dates.dt.tz_localize(None).dt.normalize()


def _check_frame(frame, *, labeled=False):
    required = ["Date", "Ticker"] + FEATURES + (["target"] if labeled else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Model table is missing these columns: {missing}")
    result = frame.copy().reset_index(drop=True)
    result["Date"] = _dates(result["Date"])
    if result["Date"].isna().any() or result["Ticker"].isna().any():
        raise ValueError("Model rows need valid Date and Ticker keys.")
    if result.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Model rows need one record per date and ticker.")
    numeric = FEATURES + (["target"] if labeled else [])
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Model features and training targets must be finite.")
    return result


def _daily_scores(values, dates):
    """Centered percentiles in [-1, 1]; tied/constant values stay tied."""
    series = pd.Series(np.asarray(values, dtype=float))
    keys = pd.Series(np.asarray(dates))
    ranks = series.groupby(keys, sort=False).rank(method="average")
    counts = series.groupby(keys, sort=False).transform("count")
    return ((ranks - 0.5) / counts - 0.5).to_numpy() * 2.0


def make_features(frame):
    """Use exactly nine raw features plus their target-free daily ranks.

    Supply the complete eligible cross-section for each date at inference, as
    changing the set of stocks changes a percentile's reference population.
    Adding a different date or changing labels cannot affect these features.
    """
    clean = _check_frame(frame)
    result = clean[FEATURES].astype(np.float32).copy()
    for name in FEATURES:
        result[f"{name}__csrank"] = _daily_scores(clean[name], clean["Date"]).astype(np.float32)
    return result[FEATURE_NAMES]


def make_rank_groups(frame, *, grades=10):
    """Sort rows by date/ticker and build date-sized LambdaMART queries.

    Each integer relevance grade describes a within-day return decile. Label
    ties share the same grade; ticker sorting never breaks return ties.
    """
    if grades < 2 or grades > 32:
        raise ValueError("Use between 2 and 32 integer relevance grades.")
    clean = _check_frame(frame, labeled=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    ranks = clean.groupby("Date", sort=False)["target"].rank(method="average")
    counts = clean.groupby("Date", sort=False)["target"].transform("count")
    labels = np.floor(grades * (ranks - 0.5) / counts).clip(0, grades - 1).to_numpy(dtype=np.int32)
    groups = clean.groupby("Date", sort=False).size().to_numpy(dtype=np.uint32)
    return clean, labels, groups


def daily_rank_ic(frame, score):
    """Equal-weight daily Spearman correlation with undefined days counted."""
    values = frame[["Date", "target"]].reset_index(drop=True).copy()
    values["score"] = np.asarray(score, dtype=float)
    values = values.replace([np.inf, -np.inf], np.nan)
    days_total = int(values["Date"].nunique())
    values = values.dropna(subset=["target", "score"])
    grouped = values.groupby("Date", sort=False)
    values["x"] = grouped["score"].rank(method="average")
    values["y"] = grouped["target"].rank(method="average")
    values["xx"] = values["x"] ** 2
    values["yy"] = values["y"] ** 2
    values["xy"] = values["x"] * values["y"]
    sums = values.groupby("Date", sort=False).agg(
        n=("x", "count"), x=("x", "sum"), y=("y", "sum"),
        xx=("xx", "sum"), yy=("yy", "sum"), xy=("xy", "sum"),
    )
    cov = sums["xy"] - sums["x"] * sums["y"] / sums["n"]
    vx = sums["xx"] - sums["x"] ** 2 / sums["n"]
    vy = sums["yy"] - sums["y"] ** 2 / sums["n"]
    valid = sums["n"].ge(3) & vx.gt(0) & vy.gt(0)
    daily = (cov.loc[valid] / np.sqrt(vx.loc[valid] * vy.loc[valid])).clip(-1, 1)
    return {
        "mean": float(daily.mean()) if len(daily) else None,
        "scored_days": int(len(daily)),
        "undefined_days": days_total - int(len(daily)),
        "total_days": days_total,
    }


def select_blend(validation, scores):
    """Choose among 12 fixed nonnegative blends using validation rankIC only.

    Single-component candidates are included, so a blend is never required.
    Ties keep the first candidate, favoring a simpler single model.
    """
    candidates = [{name: 1.0} for name in COMPONENTS]
    candidates.append({name: 0.25 for name in COMPONENTS})
    candidates.append({name: 1 / 3 for name in COMPONENTS[1:]})
    candidates.extend({left: 0.5, right: 0.5} for left, right in combinations(COMPONENTS, 2))
    trials, best_weights, best_value = [], None, -np.inf
    for weights in candidates:
        blended = sum(weight * np.asarray(scores[name]) for name, weight in weights.items())
        stats = daily_rank_ic(validation, blended)
        trials.append({"weights": weights, "validation_rank_ic": stats})
        value = stats["mean"] if stats["mean"] is not None else -np.inf
        if best_weights is None or value > best_value + 1e-12:
            best_weights, best_value = weights.copy(), value
    return best_weights, trials


def fit_return_calibration(validation, score):
    """Fit an affine score-to-return map with nonnegative slope on 2023 only.

    Each date receives equal total weight. A negative validation slope becomes
    zero; we never reverse a selected ranking using test returns. Calibration
    is an estimate of mean return, not a confidence interval or trading claim.
    """
    x = np.asarray(score, dtype=float)
    y = validation["target"].to_numpy(dtype=float)
    counts = validation.groupby("Date", sort=False)["target"].transform("count").to_numpy()
    weights = 1.0 / counts
    weights /= weights.sum()
    x_mean, y_mean = float(weights @ x), float(weights @ y)
    variance = float(weights @ ((x - x_mean) ** 2))
    slope = max(0.0, float(weights @ ((x - x_mean) * (y - y_mean))) / variance) if variance > 0 else 0.0
    return {"intercept": y_mean - slope * x_mean, "slope": slope}


@dataclass
class ShowcaseBundle:
    """Fitted components, validation-selected weights, and return calibration."""

    models: dict
    weights: dict
    return_calibration: dict
    training_metadata: dict = field(default_factory=dict)

    def predict_all(self, frame):
        """Predict without reading any target, target_rank, or later-date field."""
        clean = _check_frame(frame)
        result = clean[["Date", "Ticker"]].copy()
        if clean.empty:
            for name in COMPONENTS + ("ensemble", "ensemble_predicted_return", "reversal", "momentum"):
                result[name] = pd.Series(dtype=float)
            return result
        features = make_features(clean)
        for name in COMPONENTS:
            raw = self.models[name].predict(features)
            result[name] = _daily_scores(raw, clean["Date"])
        result["ensemble"] = sum(weight * result[name] for name, weight in self.weights.items())
        result["ensemble_predicted_return"] = (
            self.return_calibration["intercept"] + self.return_calibration["slope"] * result["ensemble"]
        )
        result["reversal"] = _daily_scores(-clean["ret_1d"], clean["Date"])
        result["momentum"] = _daily_scores(clean["mom_20d"], clean["Date"])
        return result

    def save(self, path):
        """Save a joblib bundle, metadata JSON, and native boosted-tree files."""
        path = Path(path)
        if path.suffix:
            bundle_path = path
            native_dir = path.parent / f"{path.stem}_native"
        else:
            native_dir = path
            bundle_path = path / "bundle.joblib"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        native_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, bundle_path)
        for name in ("xgb_regression", "xgb_ranker"):
            self.models[name].save_model(str(native_dir / f"{name}.ubj"))
        self.models["catboost"].save_model(str(native_dir / "catboost.cbm"))
        (native_dir / "training_metadata.json").write_text(json.dumps(self.training_metadata, indent=2) + "\n")
        return bundle_path


def _recency_weights(frame, *, group_weights=False):
    """Three-year half-life, equal aggregate weight for equally recent days."""
    dates = frame["Date"]
    if group_weights:
        dates = pd.Series(dates.drop_duplicates().to_numpy())
    ages = (frame["Date"].max() - dates).dt.days.to_numpy(dtype=float)
    weights = np.exp2(-ages / (365.25 * 3))
    if not group_weights:
        counts = frame.groupby("Date", sort=False)["target"].transform("count").to_numpy(dtype=float)
        weights /= counts
    return weights / weights.mean()


def _check_chronology(train, validation):
    if train.empty or validation.empty:
        raise ValueError("Training and validation tables must both contain rows.")
    if train["Date"].max() > TRAIN_END:
        raise ValueError("Training is frozen through 2022; later dates are not allowed.")
    if validation["Date"].min() <= TRAIN_END or validation["Date"].max() > VALIDATION_END:
        raise ValueError("Use only 2023 for stopping, blend selection, and calibration.")
    for name, frame, cutoff in (("train", train, TRAIN_END), ("validation", validation, VALIDATION_END)):
        if "LabelEndDate" in frame:
            # Unlabeled stocks still provide today's observable cross-section;
            # they have no supervised label that could cross the cutoff.
            labeled = frame.loc[frame["target"].notna()]
            ends = _dates(labeled["LabelEndDate"])
            if ends.isna().any() or ends.gt(cutoff).any():
                raise ValueError(f"Purge {name} rows whose return label crosses the period cutoff.")


def _training_context(frame):
    """Retain all observable stocks before selecting known supervised labels."""
    clean = _check_frame(frame)
    if "target" not in clean:
        raise ValueError("Training and validation context both need a target column.")
    clean["target"] = pd.to_numeric(clean["target"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return clean.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def train_showcase(train, validation, seed=42, threads=4, max_iterations=500):
    """Fit four models once, then choose and calibrate using validation only.

    Regressors predict continuous centered daily return ranks to reduce the
    effect of extreme returns. LambdaMART predicts ten relevance grades.
    There is one fixed configuration per model, no random cross-validation,
    no post-test model search, and no refit on validation after selection.
    """
    if threads < 1 or max_iterations < 1:
        raise ValueError("threads and max_iterations must both be positive.")
    train_context, validation_context = _training_context(train), _training_context(validation)
    _check_chronology(train_context, validation_context)
    train_known, validation_known = train_context["target"].notna(), validation_context["target"].notna()
    if not train_known.any() or not validation_known.any():
        raise ValueError("Training and validation both need some known supervised targets.")
    # Rank predictors before removing stocks whose future return is missing.
    # Which returns eventually become available cannot change today's input.
    x_train = make_features(train_context).loc[train_known].reset_index(drop=True)
    x_validation = make_features(validation_context).loc[validation_known].reset_index(drop=True)
    train, rank_labels, groups = make_rank_groups(train_context.loc[train_known])
    validation, validation_grades, validation_groups = make_rank_groups(validation_context.loc[validation_known])
    y_train = _daily_scores(train["target"], train["Date"])
    y_validation = _daily_scores(validation["target"], validation["Date"])
    weights = _recency_weights(train)
    stopping = min(60, max(5, max_iterations // 5))
    shared_xgb = dict(
        n_estimators=int(max_iterations), learning_rate=0.035, max_depth=4,
        min_child_weight=35, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=20.0, reg_alpha=0.1, max_bin=128, tree_method="hist",
        device="cpu", n_jobs=int(threads), random_state=int(seed),
        early_stopping_rounds=stopping,
    )
    params = {
        "elastic_net": {"alpha": 0.001, "l1_ratio": 0.15, "max_iter": 3000, "tol": 1e-5, "selection": "cyclic"},
        "xgb_regression": dict(shared_xgb, objective="reg:squarederror", eval_metric="rmse"),
        "xgb_ranker": dict(shared_xgb, objective="rank:ndcg", eval_metric="ndcg@50", ndcg_exp_gain=False,
                           lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8),
        "catboost": dict(iterations=int(max_iterations), learning_rate=0.035, depth=6, l2_leaf_reg=20.0,
                         loss_function="RMSE", eval_metric="RMSE", random_seed=int(seed),
                         thread_count=int(threads), allow_writing_files=False, verbose=False),
    }
    models, timings = {}, {}
    started_all = perf_counter()
    started = perf_counter()
    models["elastic_net"] = make_pipeline(StandardScaler(), ElasticNet(**params["elastic_net"]))
    models["elastic_net"].fit(x_train, y_train, elasticnet__sample_weight=weights)
    timings["elastic_net"] = perf_counter() - started
    started = perf_counter()
    models["xgb_regression"] = XGBRegressor(**params["xgb_regression"])
    models["xgb_regression"].fit(x_train, y_train, sample_weight=weights,
                                 eval_set=[(x_validation, y_validation)], verbose=False)
    timings["xgb_regression"] = perf_counter() - started
    started = perf_counter()
    models["xgb_ranker"] = XGBRanker(**params["xgb_ranker"])
    models["xgb_ranker"].fit(x_train, rank_labels, group=groups,
                             sample_weight=_recency_weights(train, group_weights=True),
                             eval_set=[(x_validation, validation_grades)], eval_group=[validation_groups], verbose=False)
    timings["xgb_ranker"] = perf_counter() - started
    started = perf_counter()
    models["catboost"] = CatBoostRegressor(**params["catboost"])
    models["catboost"].fit(x_train, y_train, sample_weight=weights,
                           eval_set=(x_validation, y_validation), early_stopping_rounds=stopping, use_best_model=True)
    timings["catboost"] = perf_counter() - started
    bundle = ShowcaseBundle(models, {"elastic_net": 1.0}, {"intercept": 0.0, "slope": 0.0})
    # Normalize every observable validation stock first, then score only the
    # rows with known outcomes. The same convention is used at inference.
    validation_scores = bundle.predict_all(validation_context).loc[validation_known].reset_index(drop=True)
    bundle.weights, trials = select_blend(validation, validation_scores)
    blended = sum(weight * validation_scores[name].to_numpy() for name, weight in bundle.weights.items())
    bundle.return_calibration = fit_return_calibration(validation, blended)
    metadata = {
        "seed": int(seed), "threads": int(threads), "feature_names": FEATURE_NAMES,
        "original_features": list(FEATURES), "excluded_predictors": ["Ticker", "Date", "Sector", "target", "target_rank", "LabelEndDate", "Split"],
        "candidate_params": params, "fitting_seconds": timings,
        "total_training_seconds": perf_counter() - started_all,
        "train_rows": int(len(train)), "validation_rows": int(len(validation)),
        "train_context_rows": int(len(train_context)), "validation_context_rows": int(len(validation_context)),
        "unlabeled_context_rows": {"train": int((~train_known).sum()), "validation": int((~validation_known).sum())},
        "train_dates": {"start": str(train["Date"].min().date()), "end": str(train["Date"].max().date())},
        "validation_dates": {"start": str(validation["Date"].min().date()), "end": str(validation["Date"].max().date())},
        "fit_cutoff": "2022-12-31", "selection_cutoff": "2023-12-31", "refit_on_validation": False,
        "regression_target": "same-date centered percentile rank of next-day decimal return, in [-1, 1]",
        "ranking_target": "same-date return decile relevance grade 0 through 9; equal returns stay tied",
        "recency_weighting": "three-year half-life; equal aggregate weight for equally recent dates",
        "validation_rank_ic": {name: daily_rank_ic(validation, validation_scores[name]) for name in COMPONENTS},
        "blend_trials": trials, "selected_weights": bundle.weights,
        "ensemble_validation_rank_ic": daily_rank_ic(validation, blended),
        "return_calibration": dict(bundle.return_calibration, fit_period="2023 validation only", output_unit="decimal next-day return", positive_slope=bool(bundle.return_calibration["slope"] > 0)),
        "output_units": {**{name: "same-date centered percentile score in [-1, 1]" for name in COMPONENTS},
                         "ensemble": "convex blend of daily percentile scores in [-1, 1]; not a return",
                         "ensemble_predicted_return": "validation-calibrated decimal next-day return estimate",
                         "reversal": "same-date centered percentile of minus ret_1d", "momentum": "same-date centered percentile of mom_20d"},
        "best_iterations": {
            "xgb_regression": int(models["xgb_regression"].best_iteration),
            "xgb_ranker": int(models["xgb_ranker"].best_iteration),
            "catboost": int(models["catboost"].get_best_iteration()),
        },
        "test_labels_used": False,
        "prediction_context": "Supply every eligible stock for a signal date; same-day ranks do not read labels or future dates.",
        "training_context": "Feature percentiles and validation prediction percentiles use the full observable cross-section before selecting known targets.",
        "limitations": ["2023 is used for both early stopping and blend selection, so validation results are optimistic.",
                        "Return calibration is a rough affine mean estimate and has no guarantee of realized profit.",
                        "The final test period must be evaluated once and not used to revise this bundle."],
    }
    bundle.training_metadata = metadata
    return bundle, metadata
