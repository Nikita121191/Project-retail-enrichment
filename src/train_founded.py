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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_contract import (
    CATEGORICAL_FEATURES,
    FEATURE_CONTRACT_VERSION,
    INFERENCE_SAFE_FEATURES,
    MAX_FOUNDED_YEAR,
    MIN_FOUNDED_YEAR,
    NUMERIC_FEATURES,
    validate_feature_contract,
)
from preprocessing import ensure_required_columns, prepare_features

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RANDOM_STATE = 42
CV_FOLDS = 5
DEFAULT_N_JOBS = 2

FEATURE_COLUMNS = INFERENCE_SAFE_FEATURES.copy()

REQUIRED_COLUMNS = [
    "founded",
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


def rmse(
    y_true,
    y_pred,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
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
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path
    )
    df = prepare_features(df)

    ensure_required_columns(
        df,
        REQUIRED_COLUMNS,
    )

    df["founded"] = pd.to_numeric(
        df["founded"],
        errors="coerce",
    )

    df = df[
        df["founded"].notna()
    ].copy()

    df = df[
        df["founded"].between(
            MIN_FOUNDED_YEAR,
            MAX_FOUNDED_YEAR,
        )
    ].copy()

    if len(df) < 2:
        raise ValueError(
            "Not enough valid rows for "
            "founded regression."
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

    return df


def make_ttr(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_candidates() -> list[dict]:
    candidates = [
        {
            "name": "DummyMedian",
            "family": "baseline",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DummyRegressor(
                        strategy="median",
                    ),
                ),
            ]),
            "param_grid": None,
        },
        {
            "name": "Ridge",
            "family": "linear",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    make_ttr(
                        Ridge(
                            random_state=RANDOM_STATE,
                        )
                    ),
                ),
            ]),
            "param_grid": {
                "model__regressor__alpha": [
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
                    make_ttr(
                        RandomForestRegressor(
                            n_estimators=400,
                            n_jobs=1,
                            random_state=RANDOM_STATE,
                        )
                    ),
                ),
            ]),
            "param_grid": {
                "model__regressor__max_depth": [
                    None,
                    12,
                ],
                "model__regressor__min_samples_leaf": [
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
                        make_ttr(
                            XGBRegressor(
                                n_estimators=300,
                                learning_rate=0.1,
                                max_depth=4,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                reg_lambda=1.0,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                                objective="reg:squarederror",
                                eval_metric="rmse",
                            )
                        ),
                    ),
                ]),
                "param_grid": {
                    "model__regressor__learning_rate": [
                        0.05,
                        0.1,
                    ],
                    "model__regressor__max_depth": [
                        4,
                        6,
                    ],
                },
            }
        )

    return candidates


def fit_candidate(
    candidate: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
    n_jobs: int,
) -> dict:
    """
    Candidate selection uses CV on the training split only.
    The holdout is intentionally invisible here.
    """
    pipeline = clone(
        candidate["pipeline"]
    )

    if candidate["param_grid"]:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=candidate["param_grid"],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(
            X_train,
            y_train,
        )

        fitted_model = (
            search.best_estimator_
        )
        best_params = (
            search.best_params_
        )
        cv_mean_mae = float(
            -search.best_score_
        )
        cv_std_mae = float(
            search.cv_results_[
                "std_test_score"
            ][search.best_index_]
        )
    else:
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring=(
                "neg_mean_absolute_error"
            ),
            cv=cv,
            n_jobs=n_jobs,
            error_score="raise",
        )

        mae_scores = -scores
        cv_mean_mae = float(
            np.mean(mae_scores)
        )
        cv_std_mae = float(
            np.std(mae_scores)
        )

        fitted_model = pipeline.fit(
            X_train,
            y_train,
        )
        best_params = {}

    return {
        "name": candidate["name"],
        "family": candidate["family"],
        "model": fitted_model,
        "best_params": best_params,
        "cv_best_mae": cv_mean_mae,
        "cv_std_mae": cv_std_mae,
    }


def clip_founded(
    values,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            values,
            dtype=float,
        ),
        MIN_FOUNDED_YEAR,
        MAX_FOUNDED_YEAR,
    )


def evaluate_on_holdout(
    result: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = clip_founded(
        result["model"].predict(
            X_test
        )
    )

    y_true = np.asarray(
        y_test,
        dtype=float,
    )
    abs_error = np.abs(
        y_true - y_pred
    )

    prediction_preview = (
        pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": abs_error,
            "signed_error": (
                y_pred - y_true
            ),
        })
        .sort_values(
            by="abs_error",
            ascending=False,
        )
    )

    return {
        "holdout_mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "holdout_rmse": rmse(
            y_true,
            y_pred,
        ),
        "holdout_r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "holdout_median_absolute_error": float(
            np.median(
                abs_error
            )
        ),
        "holdout_p90_absolute_error": float(
            np.quantile(
                abs_error,
                0.90,
            )
        ),
        "prediction_preview": (
            prediction_preview
        ),
    }


def save_artifacts(
    out_dir: Path,
    winner: dict,
    baseline: dict,
    leaderboard: pd.DataFrame,
    rows_used: int,
    train_rows: int,
    holdout_rows: int,
    test_size: float,
    dataset_sha256: str,
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
        / "founded_model.joblib",
    )

    winner[
        "prediction_preview"
    ].head(200).to_csv(
        out_dir
        / "founded_predictions_preview.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_improvement_abs = (
        baseline["cv_best_mae"]
        - winner["cv_best_mae"]
    )

    baseline_improvement_pct = (
        baseline_improvement_abs
        / baseline["cv_best_mae"]
        if baseline["cv_best_mae"] > 0
        else None
    )

    report = {
        "target": "founded",
        "feature_contract_version": (
            FEATURE_CONTRACT_VERSION
        ),
        "selection_metric": "cv_best_mae",
        "selection_protocol": (
            "model and hyperparameters selected by "
            "cross-validation on the training split; "
            "holdout used once for final evaluation"
        ),
        "winner_name": winner["name"],
        "winner_family": winner["family"],
        "rows_used": int(rows_used),
        "train_rows": int(train_rows),
        "holdout_rows": int(
            holdout_rows
        ),
        "test_size": float(test_size),
        "valid_target_range": [
            MIN_FOUNDED_YEAR,
            MAX_FOUNDED_YEAR,
        ],
        "feature_columns": (
            FEATURE_COLUMNS
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
        "baseline_cv_mae": (
            baseline["cv_best_mae"]
        ),
        "cv_best_mae": (
            winner["cv_best_mae"]
        ),
        "cv_std_mae": (
            winner["cv_std_mae"]
        ),
        "cv_mae_improvement_over_baseline_abs": float(
            baseline_improvement_abs
        ),
        "cv_mae_improvement_over_baseline_pct": (
            float(
                baseline_improvement_pct
            )
            if baseline_improvement_pct
            is not None
            else None
        ),
        "holdout_mae": (
            winner["holdout_mae"]
        ),
        "holdout_rmse": (
            winner["holdout_rmse"]
        ),
        "holdout_r2": (
            winner["holdout_r2"]
        ),
        "holdout_median_absolute_error": (
            winner[
                "holdout_median_absolute_error"
            ]
        ),
        "holdout_p90_absolute_error": (
            winner[
                "holdout_p90_absolute_error"
            ]
        ),
        "generalization_gap_mae": float(
            winner["holdout_mae"]
            - winner["cv_best_mae"]
        ),
        "runtime": runtime_metadata(),
    }

    with (
        out_dir
        / "founded_report.json"
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
            "founded-year regressor."
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

    validate_feature_contract(
        FEATURE_COLUMNS
    )

    dataset_sha256 = sha256_file(
        args.csv_path
    )
    df = load_and_prepare_data(
        args.csv_path
    )

    X = df[
        FEATURE_COLUMNS
    ].copy()
    y = df["founded"].astype(float)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
        )
    )

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    candidates = build_candidates()
    all_results = []

    for candidate in candidates:
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
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            n_jobs=args.n_jobs,
        )
        all_results.append(
            result
        )

        print(
            "CV MAE: "
            f"{result['cv_best_mae']:.4f} "
            "± "
            f"{result['cv_std_mae']:.4f}"
        )

    leaderboard = pd.DataFrame([
        {
            "name": result["name"],
            "family": result["family"],
            "cv_best_mae": (
                result["cv_best_mae"]
            ),
            "cv_std_mae": (
                result["cv_std_mae"]
            ),
            "best_params": json.dumps(
                result["best_params"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for result in all_results
    ]).sort_values(
        by=[
            "cv_best_mae",
            "cv_std_mae",
        ],
        ascending=[True, True],
    ).reset_index(drop=True)

    non_baseline = [
        result
        for result in all_results
        if result["family"]
        != "baseline"
    ]

    winner = min(
        non_baseline,
        key=lambda result: (
            result["cv_best_mae"],
            result["cv_std_mae"],
        ),
    )

    baseline = next(
        result
        for result in all_results
        if result["family"]
        == "baseline"
    )

    winner.update(
        evaluate_on_holdout(
            winner,
            X_test,
            y_test,
        )
    )

    print(
        "\n=============================="
    )
    print(
        "MODEL SELECTION LEADERBOARD "
        "(CV ONLY)"
    )
    print(
        "=============================="
    )
    print(leaderboard)

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
        f"Winner              : "
        f"{winner['name']}"
    )
    print(
        f"CV MAE              : "
        f"{winner['cv_best_mae']:.4f}"
    )
    print(
        f"Holdout MAE         : "
        f"{winner['holdout_mae']:.4f}"
    )
    print(
        f"Holdout RMSE        : "
        f"{winner['holdout_rmse']:.4f}"
    )
    print(
        f"Holdout R2          : "
        f"{winner['holdout_r2']:.4f}"
    )
    print(
        f"Median abs error    : "
        f"{winner['holdout_median_absolute_error']:.4f}"
    )
    print(
        f"P90 abs error       : "
        f"{winner['holdout_p90_absolute_error']:.4f}"
    )

    save_artifacts(
        out_dir=args.out_dir,
        winner=winner,
        baseline=baseline,
        leaderboard=leaderboard,
        rows_used=len(df),
        train_rows=len(X_train),
        holdout_rows=len(X_test),
        test_size=args.test_size,
        dataset_sha256=dataset_sha256,
    )

    print(
        f"\nArtifacts saved to: "
        f"{args.out_dir}"
    )


if __name__ == "__main__":
    main()
