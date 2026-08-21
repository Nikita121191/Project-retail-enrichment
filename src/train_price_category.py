from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

from model_contract import (
    CATEGORICAL_FEATURES,
    FEATURE_CONTRACT_VERSION,
    INFERENCE_SAFE_FEATURES,
    KNOWN_PRICE_LABELS,
    NUMERIC_FEATURES,
    UNKNOWN_PRICE_LABEL,
    split_price_target,
    validate_feature_contract,
)
from preprocessing import ensure_required_columns, prepare_features

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RANDOM_STATE = 42
CV_FOLDS = 5
DEFAULT_N_JOBS = 2
THRESHOLD_GRID = np.arange(0.20, 0.81, 0.05)

FEATURE_COLUMNS = INFERENCE_SAFE_FEATURES.copy()

REQUIRED_COLUMNS = [
    "price_category",
    *FEATURE_COLUMNS,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def runtime_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "platform": platform.platform(),
    }


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def build_preprocessor() -> ColumnTransformer:
    validate_feature_contract(
        FEATURE_COLUMNS
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value=0,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="Неизвестно",
                        ),
                    ),
                    ("ohe", make_ohe()),
                ]),
                CATEGORICAL_FEATURES,
            ),
            (
                "desc_tfidf",
                TfidfVectorizer(
                    max_features=4000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
                "description",
            ),
            (
                "name_tfidf",
                TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
                "name_clean",
            ),
        ],
        remainder="drop",
    )


def load_and_prepare_data(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    df = prepare_features(df)

    ensure_required_columns(
        df,
        REQUIRED_COLUMNS,
    )

    parsed = df[
        "price_category"
    ].apply(split_price_target)

    df["_price_labels"] = parsed.apply(
        lambda item: item[0]
    )
    df["_unexpected_price_labels"] = (
        parsed.apply(
            lambda item: item[1]
        )
    )

    unexpected = sorted({
        label
        for labels in df[
            "_unexpected_price_labels"
        ]
        for label in labels
    })

    if unexpected:
        raise ValueError(
            "Unexpected price_category labels found: "
            f"{unexpected}. Update the label contract "
            "or clean the source data before training."
        )

    rows_before = len(df)

    usable_mask = df[
        "_price_labels"
    ].map(bool)

    rows_unknown_only = int(
        (~usable_mask).sum()
    )

    df = df[
        usable_mask
    ].copy()

    if len(df) == 0:
        raise ValueError(
            "No rows with known price_category "
            "labels remain after excluding "
            f"'{UNKNOWN_PRICE_LABEL}'."
        )

    df["description"] = (
        df["description"]
        .fillna("")
        .astype(str)
    )
    df["name_clean"] = (
        df["name_clean"]
        .fillna("")
        .astype(str)
    )
    df["country_origin"] = (
        df["country_origin"]
        .fillna("Неизвестно")
        .astype(str)
    )

    metadata = {
        "rows_before_target_filter": int(
            rows_before
        ),
        "rows_unknown_only_dropped": (
            rows_unknown_only
        ),
        "rows_after_target_filter": int(
            len(df)
        ),
        "unexpected_labels": unexpected,
    }

    return df, metadata


def make_multilabel_target(
    label_lists: pd.Series,
) -> tuple[MultiLabelBinarizer, np.ndarray]:
    mlb = MultiLabelBinarizer(
        classes=KNOWN_PRICE_LABELS,
    )

    y = mlb.fit_transform(
        label_lists
    )

    return mlb, y


def build_candidates() -> list[dict]:
    candidates = [
        {
            "name": "LogisticRegression",
            "family": "linear",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    OneVsRestClassifier(
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                        n_jobs=1,
                    ),
                ),
            ]),
            "param_grid": {
                "model__estimator__C": [
                    0.3,
                    1.0,
                    3.0,
                ],
            },
        },
        {
            "name": "RandomForest",
            "family": "bagging",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    OneVsRestClassifier(
                        RandomForestClassifier(
                            n_estimators=400,
                            class_weight=(
                                "balanced_subsample"
                            ),
                            n_jobs=1,
                            random_state=RANDOM_STATE,
                        ),
                        n_jobs=1,
                    ),
                ),
            ]),
            "param_grid": {
                "model__estimator__max_depth": [
                    12,
                    20,
                ],
                "model__estimator__min_samples_leaf": [
                    1,
                    2,
                ],
            },
        },
    ]

    if HAS_XGBOOST:
        candidates.append(
            {
                "name": "XGBoost",
                "family": "boosting",
                "pipeline": Pipeline([
                    ("preprocessor", build_preprocessor()),
                    (
                        "model",
                        OneVsRestClassifier(
                            XGBClassifier(
                                n_estimators=300,
                                learning_rate=0.1,
                                max_depth=4,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                reg_lambda=1.0,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                                eval_metric="logloss",
                            ),
                            n_jobs=1,
                        ),
                    ),
                ]),
                "param_grid": {
                    "model__estimator__learning_rate": [
                        0.05,
                        0.1,
                    ],
                    "model__estimator__max_depth": [
                        4,
                        6,
                    ],
                },
            }
        )

    return candidates


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "samples_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="samples",
                zero_division=0,
            )
        ),
    }


def compute_dummy_cv_baseline(
    y: np.ndarray,
    cv: KFold,
) -> dict:
    scores = []

    for train_index, valid_index in cv.split(y):
        y_train = y[train_index]
        y_valid = y[valid_index]

        frequencies = np.mean(
            y_train,
            axis=0,
        )
        most_frequent_index = int(
            np.argmax(frequencies)
        )

        prediction = np.zeros_like(
            y_valid,
            dtype=int,
        )
        prediction[
            :,
            most_frequent_index,
        ] = 1

        scores.append(
            f1_score(
                y_valid,
                prediction,
                average="macro",
                zero_division=0,
            )
        )

    return {
        "name": "DummyMostFrequent",
        "family": "baseline",
        "cv_best_macro_f1": float(
            np.mean(scores)
        ),
        "cv_std_macro_f1": float(
            np.std(scores)
        ),
        "best_params": {},
    }


def fit_candidate(
    candidate: dict,
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    cv: KFold,
    n_jobs: int,
) -> dict:
    """
    Model and hyperparameter selection use development-set CV only.
    The final holdout is never visible here.
    """
    scorer = make_scorer(
        f1_score,
        average="macro",
        zero_division=0,
    )

    search = GridSearchCV(
        estimator=clone(
            candidate["pipeline"]
        ),
        param_grid=candidate[
            "param_grid"
        ],
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(
        X_development,
        y_development,
    )

    return {
        "name": candidate["name"],
        "family": candidate["family"],
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_best_macro_f1": float(
            search.best_score_
        ),
        "cv_std_macro_f1": float(
            search.cv_results_[
                "std_test_score"
            ][search.best_index_]
        ),
    }


def normalize_probability_output(
    probabilities,
) -> np.ndarray:
    if isinstance(
        probabilities,
        list,
    ):
        columns = []

        for probability in probabilities:
            probability = np.asarray(
                probability
            )

            if (
                probability.ndim == 2
                and probability.shape[1] == 2
            ):
                columns.append(
                    probability[:, 1]
                )
            else:
                columns.append(
                    probability.reshape(-1)
                )

        return np.column_stack(
            columns
        )

    probabilities = np.asarray(
        probabilities
    )

    if (
        probabilities.ndim == 3
        and probabilities.shape[2] == 2
    ):
        return probabilities[:, :, 1]

    return probabilities


def get_probabilities(
    fitted_model,
    X: pd.DataFrame,
) -> np.ndarray:
    if not hasattr(
        fitted_model,
        "predict_proba",
    ):
        raise TypeError(
            "Selected price model does not "
            "implement predict_proba."
        )

    return normalize_probability_output(
        fitted_model.predict_proba(X)
    )


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """
    Per-label thresholds are selected on out-of-fold development
    probabilities. Ties are broken toward 0.5 to reduce unnecessary
    threshold extremity.
    """
    thresholds = {}

    for index, label in enumerate(labels):
        target_column = y_true[:, index]

        if np.unique(
            target_column
        ).size < 2:
            thresholds[label] = 0.5
            continue

        candidates = []

        for threshold in THRESHOLD_GRID:
            prediction = (
                y_prob[:, index]
                >= threshold
            ).astype(int)

            score = f1_score(
                target_column,
                prediction,
                zero_division=0,
            )

            candidates.append(
                (
                    float(score),
                    -abs(
                        float(threshold)
                        - 0.5
                    ),
                    float(
                        np.round(
                            threshold,
                            2,
                        )
                    ),
                )
            )

        thresholds[label] = max(
            candidates
        )[2]

    return thresholds


def apply_thresholds(
    y_prob: np.ndarray,
    thresholds: dict[str, float],
    labels: list[str],
) -> np.ndarray:
    out = np.zeros_like(
        y_prob,
        dtype=int,
    )

    for index, label in enumerate(labels):
        out[:, index] = (
            y_prob[:, index]
            >= float(
                thresholds[label]
            )
        ).astype(int)

    empty_rows = np.where(
        out.sum(axis=1) == 0
    )[0]

    if len(empty_rows) > 0:
        max_index = np.argmax(
            y_prob[empty_rows],
            axis=1,
        )
        out[
            empty_rows,
            max_index,
        ] = 1

    return out


def calibrate_thresholds_oof(
    winner: dict,
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    cv: KFold,
    n_jobs: int,
    labels: list[str],
) -> dict:
    """
    Thresholds are calibrated from out-of-fold predictions produced by
    the already-selected model/hyperparameters. No in-fold fitted
    probabilities are used for threshold tuning.
    """
    oof_probabilities = cross_val_predict(
        clone(winner["model"]),
        X_development,
        y_development,
        cv=cv,
        n_jobs=n_jobs,
        method="predict_proba",
    )

    oof_probabilities = (
        normalize_probability_output(
            oof_probabilities
        )
    )

    optimized_thresholds = (
        optimize_thresholds(
            y_development,
            oof_probabilities,
            labels,
        )
    )

    optimized_prediction = (
        apply_thresholds(
            oof_probabilities,
            optimized_thresholds,
            labels,
        )
    )

    optimized_metrics = (
        multilabel_metrics(
            y_development,
            optimized_prediction,
        )
    )

    default_thresholds = {
        label: 0.5
        for label in labels
    }

    default_prediction = (
        apply_thresholds(
            oof_probabilities,
            default_thresholds,
            labels,
        )
    )

    default_metrics = (
        multilabel_metrics(
            y_development,
            default_prediction,
        )
    )

    if (
        optimized_metrics["macro_f1"]
        >= default_metrics["macro_f1"]
    ):
        selected_thresholds = (
            optimized_thresholds
        )
        selected_metrics = (
            optimized_metrics
        )
        strategy = (
            "oof_per_label_optimized"
        )
    else:
        selected_thresholds = (
            default_thresholds
        )
        selected_metrics = (
            default_metrics
        )
        strategy = "default_0.5"

    return {
        "thresholds": (
            selected_thresholds
        ),
        "threshold_strategy": strategy,
        "oof_default_metrics": (
            default_metrics
        ),
        "oof_selected_metrics": (
            selected_metrics
        ),
        "oof_optimized_metrics": (
            optimized_metrics
        ),
    }


def evaluate_on_holdout(
    winner: dict,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    labels: list[str],
) -> dict:
    direct_prediction = (
        winner["model"].predict(
            X_holdout
        )
    )
    direct_metrics = (
        multilabel_metrics(
            y_holdout,
            direct_prediction,
        )
    )

    probabilities = get_probabilities(
        winner["model"],
        X_holdout,
    )

    serving_prediction = (
        apply_thresholds(
            probabilities,
            winner["thresholds"],
            labels,
        )
    )

    serving_metrics = (
        multilabel_metrics(
            y_holdout,
            serving_prediction,
        )
    )

    report_dict = (
        classification_report(
            y_holdout,
            serving_prediction,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )
    )

    report_text = (
        classification_report(
            y_holdout,
            serving_prediction,
            target_names=labels,
            zero_division=0,
        )
    )

    return {
        "holdout_direct_macro_f1": (
            direct_metrics["macro_f1"]
        ),
        "holdout_direct_micro_f1": (
            direct_metrics["micro_f1"]
        ),
        "holdout_serving_macro_f1": (
            serving_metrics["macro_f1"]
        ),
        "holdout_serving_micro_f1": (
            serving_metrics["micro_f1"]
        ),
        "holdout_serving_weighted_f1": (
            serving_metrics[
                "weighted_f1"
            ]
        ),
        "holdout_serving_samples_f1": (
            serving_metrics[
                "samples_f1"
            ]
        ),
        "classification_report_text": (
            report_text
        ),
        "classification_report_df": (
            pd.DataFrame(
                report_dict
            ).T
        ),
    }


def compute_holdout_dummy_metrics(
    y_development: np.ndarray,
    y_holdout: np.ndarray,
) -> dict:
    frequencies = np.mean(
        y_development,
        axis=0,
    )
    most_frequent_index = int(
        np.argmax(frequencies)
    )

    prediction = np.zeros_like(
        y_holdout,
        dtype=int,
    )
    prediction[
        :,
        most_frequent_index,
    ] = 1

    return multilabel_metrics(
        y_holdout,
        prediction,
    )


def validate_label_support(
    y: np.ndarray,
    labels: list[str],
) -> dict[str, int]:
    supports = {
        label: int(
            y[:, index].sum()
        )
        for index, label
        in enumerate(labels)
    }

    too_rare = {
        label: support
        for label, support
        in supports.items()
        if support < CV_FOLDS
    }

    if too_rare:
        raise ValueError(
            "Some price labels have fewer "
            f"than {CV_FOLDS} positive rows: "
            f"{too_rare}"
        )

    return supports


def save_artifacts(
    out_dir: Path,
    winner: dict,
    baseline: dict,
    leaderboard: pd.DataFrame,
    labels: list[str],
    rows_used: int,
    development_rows: int,
    holdout_rows: int,
    test_size: float,
    dataset_sha256: str,
    target_filter_metadata: dict,
    development_support: dict,
    holdout_support: dict,
    holdout_dummy_metrics: dict,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard.to_csv(
        out_dir / "leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        winner["model"],
        out_dir
        / "price_category_model.joblib",
    )

    with (
        out_dir
        / "price_category_thresholds.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            winner["thresholds"],
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (
        out_dir
        / "price_category_labels.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            labels,
            file,
            ensure_ascii=False,
            indent=2,
        )

    winner[
        "classification_report_df"
    ].to_csv(
        out_dir
        / "price_category_classification_report.csv",
        encoding="utf-8-sig",
    )

    with (
        out_dir
        / "price_category_classification_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            winner[
                "classification_report_text"
            ]
        )

    report = {
        "target": "price_category",
        "feature_contract_version": (
            FEATURE_CONTRACT_VERSION
        ),
        "selection_metric": (
            "cv_best_macro_f1"
        ),
        "selection_protocol": (
            "model and hyperparameters selected by "
            "cross-validation on the development set; "
            "thresholds calibrated from out-of-fold "
            "development probabilities; holdout used "
            "once for final evaluation"
        ),
        "threshold_calibration": (
            "development_oof_cv"
        ),
        "threshold_strategy": (
            winner[
                "threshold_strategy"
            ]
        ),
        "winner_name": winner["name"],
        "winner_family": (
            winner["family"]
        ),
        "rows_used": int(rows_used),
        "development_rows": int(
            development_rows
        ),
        "holdout_rows": int(
            holdout_rows
        ),
        "test_size": float(test_size),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "labels": labels,
        "unknown_label_policy": (
            f"'{UNKNOWN_PRICE_LABEL}' is treated "
            "as missing target information and "
            "is excluded from training labels"
        ),
        "target_filter": (
            target_filter_metadata
        ),
        "dataset_sha256": (
            dataset_sha256
        ),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "best_params": (
            winner["best_params"]
        ),
        "baseline_name": (
            baseline["name"]
        ),
        "baseline_cv_macro_f1": (
            baseline[
                "cv_best_macro_f1"
            ]
        ),
        "baseline_cv_std_macro_f1": (
            baseline[
                "cv_std_macro_f1"
            ]
        ),
        "cv_best_macro_f1": (
            winner[
                "cv_best_macro_f1"
            ]
        ),
        "cv_std_macro_f1": (
            winner[
                "cv_std_macro_f1"
            ]
        ),
        "cv_uplift_over_baseline": float(
            winner["cv_best_macro_f1"]
            - baseline["cv_best_macro_f1"]
        ),
        "oof_default_macro_f1": (
            winner[
                "oof_default_metrics"
            ]["macro_f1"]
        ),
        "oof_selected_macro_f1": (
            winner[
                "oof_selected_metrics"
            ]["macro_f1"]
        ),
        "oof_selected_micro_f1": (
            winner[
                "oof_selected_metrics"
            ]["micro_f1"]
        ),
        "holdout_direct_macro_f1": (
            winner[
                "holdout_direct_macro_f1"
            ]
        ),
        "holdout_direct_micro_f1": (
            winner[
                "holdout_direct_micro_f1"
            ]
        ),
        "holdout_serving_macro_f1": (
            winner[
                "holdout_serving_macro_f1"
            ]
        ),
        "holdout_serving_micro_f1": (
            winner[
                "holdout_serving_micro_f1"
            ]
        ),
        "holdout_serving_weighted_f1": (
            winner[
                "holdout_serving_weighted_f1"
            ]
        ),
        "holdout_serving_samples_f1": (
            winner[
                "holdout_serving_samples_f1"
            ]
        ),
        "holdout_dummy_macro_f1": (
            holdout_dummy_metrics[
                "macro_f1"
            ]
        ),
        "development_label_support": (
            development_support
        ),
        "holdout_label_support": (
            holdout_support
        ),
        "thresholds": (
            winner["thresholds"]
        ),
        "generalization_gap_macro_f1": float(
            winner[
                "holdout_serving_macro_f1"
            ]
            - winner[
                "oof_selected_metrics"
            ]["macro_f1"]
        ),
        # Backward-compatible aliases for existing downstream readers.
        "threshold_macro_f1": (
            winner[
                "holdout_serving_macro_f1"
            ]
        ),
        "threshold_micro_f1": (
            winner[
                "holdout_serving_micro_f1"
            ]
        ),
        "threshold_weighted_f1": (
            winner[
                "holdout_serving_weighted_f1"
            ]
        ),
        "threshold_samples_f1": (
            winner[
                "holdout_serving_samples_f1"
            ]
        ),
        "runtime": runtime_metadata(),
    }

    with (
        out_dir
        / "price_category_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the "
            "multilabel price_category classifier."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )
    # Kept for backward compatibility with the current Airflow DAG.
    # Threshold calibration no longer uses a dedicated validation split.
    parser.add_argument(
        "--validation_size",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(
            f"CSV file not found: "
            f"{args.csv_path}"
        )

    if not 0.0 < args.test_size < 1.0:
        parser.error(
            "--test_size must be "
            "between 0 and 1."
        )

    if (
        args.n_jobs == 0
        or args.n_jobs < -1
    ):
        parser.error(
            "--n_jobs must be -1 or "
            "a positive integer."
        )

    if (
        args.validation_size
        is not None
    ):
        print(
            "NOTE: --validation_size is deprecated "
            "and ignored. Thresholds are calibrated "
            "from out-of-fold development predictions."
        )

    validate_feature_contract(
        FEATURE_COLUMNS
    )

    dataset_sha256 = sha256_file(
        args.csv_path
    )

    df, target_filter_metadata = (
        load_and_prepare_data(
            args.csv_path
        )
    )

    mlb, y = make_multilabel_target(
        df["_price_labels"]
    )
    labels = list(
        mlb.classes_
    )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    label_cardinality = y.sum(
        axis=1
    )

    (
        X_development,
        X_holdout,
        y_development,
        y_holdout,
    ) = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=label_cardinality,
    )

    development_support = (
        validate_label_support(
            y_development,
            labels,
        )
    )

    holdout_support = {
        label: int(
            y_holdout[:, index].sum()
        )
        for index, label
        in enumerate(labels)
    }

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    baseline = compute_dummy_cv_baseline(
        y_development,
        cv,
    )

    all_results = []

    for candidate in build_candidates():
        print(
            "\n=============================="
        )
        print(
            f"Training: "
            f"{candidate['name']} "
            f"[{candidate['family']}]"
        )
        print(
            "=============================="
        )

        result = fit_candidate(
            candidate=candidate,
            X_development=(
                X_development
            ),
            y_development=(
                y_development
            ),
            cv=cv,
            n_jobs=args.n_jobs,
        )

        all_results.append(
            result
        )

        print(
            "CV Macro F1: "
            f"{result['cv_best_macro_f1']:.4f} "
            "± "
            f"{result['cv_std_macro_f1']:.4f}"
        )

    winner = max(
        all_results,
        key=lambda result: (
            result[
                "cv_best_macro_f1"
            ],
            -result[
                "cv_std_macro_f1"
            ],
        ),
    )

    threshold_result = (
        calibrate_thresholds_oof(
            winner=winner,
            X_development=(
                X_development
            ),
            y_development=(
                y_development
            ),
            cv=cv,
            n_jobs=args.n_jobs,
            labels=labels,
        )
    )
    winner.update(
        threshold_result
    )

    winner.update(
        evaluate_on_holdout(
            winner=winner,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            labels=labels,
        )
    )

    holdout_dummy_metrics = (
        compute_holdout_dummy_metrics(
            y_development,
            y_holdout,
        )
    )

    leaderboard_rows = [
        {
            "name": baseline["name"],
            "family": (
                baseline["family"]
            ),
            "cv_best_macro_f1": (
                baseline[
                    "cv_best_macro_f1"
                ]
            ),
            "cv_std_macro_f1": (
                baseline[
                    "cv_std_macro_f1"
                ]
            ),
            "best_params": "{}",
        }
    ]

    leaderboard_rows.extend([
        {
            "name": result["name"],
            "family": result["family"],
            "cv_best_macro_f1": (
                result[
                    "cv_best_macro_f1"
                ]
            ),
            "cv_std_macro_f1": (
                result[
                    "cv_std_macro_f1"
                ]
            ),
            "best_params": json.dumps(
                result["best_params"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for result in all_results
    ])

    leaderboard = (
        pd.DataFrame(
            leaderboard_rows
        )
        .sort_values(
            by=[
                "cv_best_macro_f1",
                "cv_std_macro_f1",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n=============================="
    )
    print(
        "MODEL SELECTION LEADERBOARD "
        "(DEVELOPMENT CV ONLY)"
    )
    print(
        "=============================="
    )
    print(leaderboard)

    print(
        "\n=============================="
    )
    print(
        "THRESHOLD CALIBRATION"
    )
    print(
        "=============================="
    )
    print(
        "Strategy              : "
        f"{winner['threshold_strategy']}"
    )
    print(
        "OOF default Macro F1  : "
        f"{winner['oof_default_metrics']['macro_f1']:.4f}"
    )
    print(
        "OOF serving Macro F1  : "
        f"{winner['oof_selected_metrics']['macro_f1']:.4f}"
    )

    print(
        "\n=============================="
    )
    print(
        "FINAL HOLDOUT EVALUATION"
    )
    print(
        "=============================="
    )
    print(
        f"Winner                : "
        f"{winner['name']}"
    )
    print(
        "Holdout direct Macro  : "
        f"{winner['holdout_direct_macro_f1']:.4f}"
    )
    print(
        "Holdout serving Macro : "
        f"{winner['holdout_serving_macro_f1']:.4f}"
    )
    print(
        "Holdout serving Micro : "
        f"{winner['holdout_serving_micro_f1']:.4f}"
    )
    print(
        "Holdout serving Sample: "
        f"{winner['holdout_serving_samples_f1']:.4f}"
    )

    print(
        "\nServing thresholds:"
    )
    for label in labels:
        print(
            f"  {label}: "
            f"{winner['thresholds'][label]:.2f}"
        )

    save_artifacts(
        out_dir=args.out_dir,
        winner=winner,
        baseline=baseline,
        leaderboard=leaderboard,
        labels=labels,
        rows_used=len(df),
        development_rows=(
            len(X_development)
        ),
        holdout_rows=(
            len(X_holdout)
        ),
        test_size=args.test_size,
        dataset_sha256=(
            dataset_sha256
        ),
        target_filter_metadata=(
            target_filter_metadata
        ),
        development_support=(
            development_support
        ),
        holdout_support=(
            holdout_support
        ),
        holdout_dummy_metrics=(
            holdout_dummy_metrics
        ),
    )

    print(
        f"\nArtifacts saved to: "
        f"{args.out_dir}"
    )


if __name__ == "__main__":
    main()
