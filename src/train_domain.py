from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from model_contract import (
    CATEGORICAL_FEATURES,
    FEATURE_CONTRACT_VERSION,
    INFERENCE_SAFE_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    UNKNOWN_DOMAIN,
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
MIN_CLASS_ROWS = 5
DEFAULT_N_JOBS = 2

FEATURE_COLUMNS = INFERENCE_SAFE_FEATURES.copy()

REQUIRED_COLUMNS = [
    "domain",
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
    validate_feature_contract(FEATURE_COLUMNS)

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


def load_and_prepare_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = prepare_features(df)

    ensure_required_columns(
        df,
        REQUIRED_COLUMNS,
    )

    df = df[df["domain"].notna()].copy()
    df["domain"] = (
        df["domain"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["domain"] != "")
        & (df["domain"] != UNKNOWN_DOMAIN)
    ].copy()

    class_counts = df["domain"].value_counts()
    keep_classes = class_counts[
        class_counts >= MIN_CLASS_ROWS
    ].index

    df = df[
        df["domain"].isin(keep_classes)
    ].copy()

    if df["domain"].nunique() < 2:
        raise ValueError(
            "Domain training requires at least two classes "
            f"with >= {MIN_CLASS_ROWS} rows."
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


def build_candidates(n_classes: int) -> list[dict]:
    candidates = [
        {
            "name": "DummyMostFrequent",
            "family": "baseline",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DummyClassifier(
                        strategy="most_frequent",
                    ),
                ),
            ]),
            "param_grid": None,
            "needs_label_encoding": False,
        },
        {
            "name": "LogisticRegression",
            "family": "linear",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="saga",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            "param_grid": {
                "model__C": [0.3, 1.0, 3.0],
            },
            "needs_label_encoding": False,
        },
        {
            "name": "RandomForest",
            "family": "bagging",
            "pipeline": Pipeline([
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            "param_grid": {
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 2],
            },
            "needs_label_encoding": False,
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
                        XGBClassifier(
                            objective="multi:softprob",
                            num_class=n_classes,
                            n_estimators=300,
                            learning_rate=0.1,
                            max_depth=4,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                            eval_metric="mlogloss",
                        ),
                    ),
                ]),
                "param_grid": {
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [4, 6],
                },
                "needs_label_encoding": True,
            }
        )

    return candidates


def fit_candidate(
    candidate: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
) -> dict:
    """
    Model/hyperparameter selection is restricted to the training split.
    The final holdout is never passed into this function.
    """
    pipeline = clone(
        candidate["pipeline"]
    )

    if candidate["needs_label_encoding"]:
        label_encoder = LabelEncoder()
        y_train_fit = label_encoder.fit_transform(
            y_train.astype(str)
        )
    else:
        label_encoder = None
        y_train_fit = y_train.astype(str).values

    if candidate["param_grid"]:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=candidate["param_grid"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(
            X_train,
            y_train_fit,
        )

        fitted_model = search.best_estimator_
        best_params = search.best_params_
        cv_mean = float(search.best_score_)
        cv_std = float(
            search.cv_results_["std_test_score"][
                search.best_index_
            ]
        )
    else:
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train_fit,
            scoring="f1_macro",
            cv=cv,
            n_jobs=n_jobs,
            error_score="raise",
        )

        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
        fitted_model = pipeline.fit(
            X_train,
            y_train_fit,
        )
        best_params = {}

    return {
        "name": candidate["name"],
        "family": candidate["family"],
        "model": fitted_model,
        "best_params": best_params,
        "cv_best_macro_f1": cv_mean,
        "cv_std_macro_f1": cv_std,
        "label_encoder": label_encoder,
    }


def decode_predictions(
    result: dict,
    raw_prediction,
) -> np.ndarray:
    if result["label_encoder"] is not None:
        return result[
            "label_encoder"
        ].inverse_transform(
            np.asarray(
                raw_prediction,
                dtype=int,
            )
        )

    return np.asarray(
        raw_prediction,
    ).astype(str)


def evaluate_on_holdout(
    result: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    raw_prediction = result[
        "model"
    ].predict(X_test)

    y_pred = decode_predictions(
        result,
        raw_prediction,
    )
    y_true = y_test.astype(str).values

    report_dict = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    labels = sorted(
        np.unique(
            np.concatenate(
                [y_true, y_pred]
            )
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
        "holdout_accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "holdout_balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "holdout_macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "holdout_macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "holdout_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "holdout_weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report_text": (
            classification_report(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "classification_report_df": (
            pd.DataFrame(
                report_dict
            ).T
        ),
        "confusion_matrix_df": (
            pd.DataFrame(
                cm,
                index=labels,
                columns=labels,
            )
        ),
    }


def save_artifacts(
    out_dir: Path,
    winner: dict,
    baseline: dict,
    leaderboard: pd.DataFrame,
    classes: list[str],
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
        out_dir / "domain_model.joblib",
    )

    if winner[
        "label_encoder"
    ] is not None:
        joblib.dump(
            winner["label_encoder"],
            out_dir
            / "domain_label_encoder.joblib",
        )

    winner[
        "classification_report_df"
    ].to_csv(
        out_dir
        / "domain_classification_report.csv",
        encoding="utf-8-sig",
    )

    winner[
        "confusion_matrix_df"
    ].to_csv(
        out_dir
        / "domain_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    with (
        out_dir
        / "domain_classification_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            winner[
                "classification_report_text"
            ]
        )

    cv_uplift = (
        winner["cv_best_macro_f1"]
        - baseline["cv_best_macro_f1"]
    )

    report = {
        "target": "domain",
        "feature_contract_version": (
            FEATURE_CONTRACT_VERSION
        ),
        "selection_metric": (
            "cv_best_macro_f1"
        ),
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
        "n_classes": int(len(classes)),
        "min_class_rows": (
            MIN_CLASS_ROWS
        ),
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
        "baseline_cv_macro_f1": (
            baseline[
                "cv_best_macro_f1"
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
            cv_uplift
        ),
        "holdout_accuracy": (
            winner[
                "holdout_accuracy"
            ]
        ),
        "holdout_balanced_accuracy": (
            winner[
                "holdout_balanced_accuracy"
            ]
        ),
        "holdout_macro_precision": (
            winner[
                "holdout_macro_precision"
            ]
        ),
        "holdout_macro_recall": (
            winner[
                "holdout_macro_recall"
            ]
        ),
        "holdout_macro_f1": (
            winner[
                "holdout_macro_f1"
            ]
        ),
        "holdout_weighted_f1": (
            winner[
                "holdout_weighted_f1"
            ]
        ),
        "generalization_gap_macro_f1": float(
            winner["holdout_macro_f1"]
            - winner["cv_best_macro_f1"]
        ),
        "runtime": runtime_metadata(),
    }

    with (
        out_dir
        / "domain_report.json"
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

    with (
        out_dir
        / "domain_classes.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            sorted(classes),
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the domain classifier."
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
    y = df["domain"].astype(str)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    candidates = build_candidates(
        n_classes=y.nunique(),
    )

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
            "CV Macro F1: "
            f"{result['cv_best_macro_f1']:.4f} "
            "± "
            f"{result['cv_std_macro_f1']:.4f}"
        )

    leaderboard = pd.DataFrame([
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
    ]).sort_values(
        by=[
            "cv_best_macro_f1",
            "cv_std_macro_f1",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)

    non_baseline = [
        result
        for result in all_results
        if result["family"]
        != "baseline"
    ]

    winner = max(
        non_baseline,
        key=lambda result: (
            result[
                "cv_best_macro_f1"
            ],
            -result[
                "cv_std_macro_f1"
            ],
        ),
    )

    baseline = next(
        result
        for result in all_results
        if result["family"]
        == "baseline"
    )

    holdout_metrics = (
        evaluate_on_holdout(
            winner,
            X_test,
            y_test,
        )
    )
    winner.update(
        holdout_metrics
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
        f"CV Macro F1         : "
        f"{winner['cv_best_macro_f1']:.4f}"
    )
    print(
        f"Holdout Accuracy    : "
        f"{winner['holdout_accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy   : "
        f"{winner['holdout_balanced_accuracy']:.4f}"
    )
    print(
        f"Holdout Macro F1    : "
        f"{winner['holdout_macro_f1']:.4f}"
    )
    print(
        f"Holdout Weighted F1 : "
        f"{winner['holdout_weighted_f1']:.4f}"
    )

    save_artifacts(
        out_dir=args.out_dir,
        winner=winner,
        baseline=baseline,
        leaderboard=leaderboard,
        classes=sorted(
            y.unique().tolist()
        ),
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
