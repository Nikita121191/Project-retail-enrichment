import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from preprocessing import prepare_features, ensure_required_columns

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RANDOM_STATE = 42


REQUIRED_COLUMNS = [
    "name",
    "name_clean",
    "description",
    "price_category",
    "country_origin",
    "domain",
    "presence_world",
    "presence_russia",
    "presence_own",
    "presence_franchise",
    "presence_regions",
    "n_regions",
    "plans",
    "founded",
    "desc_len",
    "desc_has_digits",
    "desc_year_mentions",
    "has_total_rented_area",
]


FEATURE_COLUMNS = [
    "description",
    "name_clean",
    "price_category",
    "country_origin",
    "founded",
    "presence_world",
    "plans",
    "presence_own",
    "presence_franchise",
    "n_regions",
    "desc_len",
    "desc_has_digits",
    "desc_year_mentions",
    "has_total_rented_area",
]


NUMERIC_FEATURES = [
    "founded",
    "presence_world",
    "plans",
    "presence_own",
    "presence_franchise",
    "n_regions",
    "desc_len",
    "desc_has_digits",
    "desc_year_mentions",
    "has_total_rented_area",
]


CATEGORICAL_FEATURES = [
    "country_origin",
    "price_category",
]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="неизвестно")),
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


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = prepare_features(df)

    ensure_required_columns(df, REQUIRED_COLUMNS)

    df = df[df["domain"].notna()].copy()
    df["domain"] = df["domain"].astype(str).str.strip()
    df = df[df["domain"] != ""].copy()
    df = df[df["domain"] != "Неизвестно"].copy()

    class_counts = df["domain"].value_counts()
    keep_classes = class_counts[class_counts >= 5].index
    df = df[df["domain"].isin(keep_classes)].copy()

    df["description"] = df["description"].fillna("").astype(str)
    df["name_clean"] = df["name_clean"].fillna("").astype(str)
    df["country_origin"] = df["country_origin"].fillna("Неизвестно").astype(str)
    df["price_category"] = df["price_category"].fillna("неизвестно").astype(str)

    return df


def build_candidates(n_classes: int):
    preprocessor = build_preprocessor()

    candidates = [
        {
            "name": "DummyMostFrequent",
            "family": "baseline",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]),
            "param_grid": None,
            "needs_label_encoding": False,
        },
        {
            "name": "LogisticRegression",
            "family": "linear",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="saga",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )),
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
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )),
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
                    ("preprocessor", preprocessor),
                    ("model", XGBClassifier(
                        objective="multi:softprob",
                        num_class=n_classes,
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="mlogloss",
                    )),
                ]),
                "param_grid": {
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [4, 6],
                },
                "needs_label_encoding": True,
            }
        )

    return candidates


def evaluate_candidate(candidate, X_train, y_train, X_test, y_test, cv):
    pipe = clone(candidate["pipeline"])

    if candidate["needs_label_encoding"]:
        le = LabelEncoder()
        y_train_fit = le.fit_transform(y_train)
        y_test_true = y_test.astype(str).values
    else:
        le = None
        y_train_fit = y_train
        y_test_true = y_test.astype(str).values

    if candidate["param_grid"]:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=candidate["param_grid"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train_fit)
        fitted_model = search.best_estimator_
        best_params = search.best_params_
        cv_best_score = float(search.best_score_)
    else:
        fitted_model = pipe.fit(X_train, y_train_fit)
        best_params = {}
        cv_best_score = None

    y_pred_raw = fitted_model.predict(X_test)

    if le is not None:
        y_pred = le.inverse_transform(y_pred_raw)
    else:
        y_pred = y_pred_raw

    acc = float(accuracy_score(y_test_true, y_pred))
    macro_f1 = float(f1_score(y_test_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test_true, y_pred, average="weighted"))
    report_text = classification_report(y_test_true, y_pred, zero_division=0)

    return {
        "name": candidate["name"],
        "family": candidate["family"],
        "model": fitted_model,
        "best_params": best_params,
        "cv_best_macro_f1": cv_best_score,
        "holdout_accuracy": acc,
        "holdout_macro_f1": macro_f1,
        "holdout_weighted_f1": weighted_f1,
        "classification_report": report_text,
        "label_encoder": le,
    }


def save_artifacts(
    out_dir: Path,
    winner: dict,
    leaderboard: pd.DataFrame,
    classes: list,
    rows_used: int,
    feature_columns: list,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    joblib.dump(winner["model"], out_dir / "domain_model.joblib")

    if winner["label_encoder"] is not None:
        joblib.dump(winner["label_encoder"], out_dir / "domain_label_encoder.joblib")

    report = {
        "target": "domain",
        "winner_name": winner["name"],
        "winner_family": winner["family"],
        "rows_used": int(rows_used),
        "n_classes": int(len(classes)),
        "feature_columns": feature_columns,
        "best_params": winner["best_params"],
        "cv_best_macro_f1": winner["cv_best_macro_f1"],
        "holdout_accuracy": winner["holdout_accuracy"],
        "holdout_macro_f1": winner["holdout_macro_f1"],
        "holdout_weighted_f1": winner["holdout_weighted_f1"],
    }

    with open(out_dir / "domain_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "domain_classes.json", "w", encoding="utf-8") as f:
        json.dump(sorted(classes), f, ensure_ascii=False, indent=2)

    with open(out_dir / "domain_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(winner["classification_report"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")

    if not 0.0 < args.test_size < 1.0:
        parser.error("--test_size must be between 0 and 1.")

    out_dir = args.out_dir
    df = load_and_prepare_data(args.csv_path)

    X = df[FEATURE_COLUMNS].copy()
    y = df["domain"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates(n_classes=y.nunique())

    all_results = []

    for candidate in candidates:
        print("\n==============================")
        print(f"Training: {candidate['name']} [{candidate['family']}]")
        print("==============================")

        result = evaluate_candidate(candidate, X_train, y_train, X_test, y_test, cv)
        all_results.append(result)

        print(f"Accuracy      : {result['holdout_accuracy']:.4f}")
        print(f"Macro F1      : {result['holdout_macro_f1']:.4f}")
        print(f"Weighted F1   : {result['holdout_weighted_f1']:.4f}")
        print(result["classification_report"])

    leaderboard = pd.DataFrame([
        {
            "name": r["name"],
            "family": r["family"],
            "cv_best_macro_f1": r["cv_best_macro_f1"],
            "holdout_macro_f1": r["holdout_macro_f1"],
            "holdout_accuracy": r["holdout_accuracy"],
            "holdout_weighted_f1": r["holdout_weighted_f1"],
        }
        for r in all_results
    ]).sort_values(
        by=["holdout_macro_f1", "holdout_accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)

    winner_name = leaderboard.iloc[0]["name"]
    winner = next(r for r in all_results if r["name"] == winner_name)

    print("\n==============================")
    print("FINAL LEADERBOARD")
    print("==============================")
    print(leaderboard[["name", "family", "holdout_macro_f1", "holdout_accuracy"]])

    winner_print = {
        "name": winner["name"],
        "family": winner["family"],
        "best_params": winner["best_params"],
        "cv_best_macro_f1": winner["cv_best_macro_f1"],
        "holdout_accuracy": winner["holdout_accuracy"],
        "holdout_macro_f1": winner["holdout_macro_f1"],
        "holdout_weighted_f1": winner["holdout_weighted_f1"],
    }

    print("\nWinner:")
    print(json.dumps(winner_print, ensure_ascii=False, indent=2))

    save_artifacts(
        out_dir=out_dir,
        winner=winner,
        leaderboard=leaderboard,
        classes=sorted(y.unique().tolist()),
        rows_used=len(df),
        feature_columns=FEATURE_COLUMNS,
    )

    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
