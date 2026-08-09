import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import prepare_features, ensure_required_columns

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RANDOM_STATE = 42
MIN_FOUNDED_YEAR = 1850
MAX_FOUNDED_YEAR = 2025


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
    "domain",
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
    "domain",
]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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

    df["founded"] = pd.to_numeric(df["founded"], errors="coerce")
    df = df[df["founded"].notna()].copy()
    df = df[df["founded"].between(MIN_FOUNDED_YEAR, MAX_FOUNDED_YEAR)].copy()

    df["description"] = df["description"].fillna("").astype(str)
    df["name_clean"] = df["name_clean"].fillna("").astype(str)
    df["country_origin"] = df["country_origin"].fillna("Неизвестно").astype(str)
    df["price_category"] = df["price_category"].fillna("неизвестно").astype(str)
    df["domain"] = df["domain"].fillna("Неизвестно").astype(str)

    return df


def make_ttr(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_candidates():
    preprocessor = build_preprocessor()

    candidates = [
        {
            "name": "DummyMedian",
            "family": "baseline",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]),
            "param_grid": None,
        },
        {
            "name": "Ridge",
            "family": "linear",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", make_ttr(Ridge(random_state=RANDOM_STATE))),
            ]),
            "param_grid": {
                "model__regressor__alpha": [0.3, 1.0, 3.0],
            },
        },
        {
            "name": "RandomForest",
            "family": "bagging",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", make_ttr(RandomForestRegressor(
                    n_estimators=400,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ))),
            ]),
            "param_grid": {
                "model__regressor__max_depth": [None, 12],
                "model__regressor__min_samples_leaf": [1, 2],
            },
        },
    ]

    if HAS_XGBOOST:
        candidates.append(
            {
                "name": "XGBoost",
                "family": "boosting",
                "pipeline": Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", make_ttr(XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        objective="reg:squarederror",
                        eval_metric="rmse",
                    ))),
                ]),
                "param_grid": {
                    "model__regressor__learning_rate": [0.05, 0.1],
                    "model__regressor__max_depth": [4, 6],
                },
            }
        )

    return candidates


def evaluate_candidate(candidate, X_train, y_train, X_test, y_test, cv):
    pipe = clone(candidate["pipeline"])

    if candidate["param_grid"]:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=candidate["param_grid"],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted_model = search.best_estimator_
        best_params = search.best_params_
        cv_best_mae = float(-search.best_score_)
    else:
        fitted_model = pipe.fit(X_train, y_train)
        best_params = {}
        cv_best_mae = None

    y_pred = fitted_model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse_value = rmse(y_test, y_pred)
    r2 = float(r2_score(y_test, y_pred))

    prediction_preview = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "abs_error": np.abs(y_test.values - y_pred),
    }).sort_values(by="abs_error", ascending=False)

    return {
        "name": candidate["name"],
        "family": candidate["family"],
        "model": fitted_model,
        "best_params": best_params,
        "cv_best_mae": cv_best_mae,
        "holdout_mae": mae,
        "holdout_rmse": rmse_value,
        "holdout_r2": r2,
        "prediction_preview": prediction_preview,
    }


def save_artifacts(
    out_dir: Path,
    winner: dict,
    leaderboard: pd.DataFrame,
    rows_used: int,
    feature_columns: list,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    joblib.dump(winner["model"], out_dir / "founded_model.joblib")

    report = {
        "target": "founded",
        "winner_name": winner["name"],
        "winner_family": winner["family"],
        "rows_used": int(rows_used),
        "feature_columns": feature_columns,
        "best_params": winner["best_params"],
        "cv_best_mae": winner["cv_best_mae"],
        "holdout_mae": winner["holdout_mae"],
        "holdout_rmse": winner["holdout_rmse"],
        "holdout_r2": winner["holdout_r2"],
    }

    with open(out_dir / "founded_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    winner["prediction_preview"].head(200).to_csv(
        out_dir / "founded_predictions_preview.csv",
        index=False,
        encoding="utf-8-sig",
    )


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
    y = df["founded"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()

    all_results = []

    for candidate in candidates:
        print("\n==============================")
        print(f"Training: {candidate['name']} [{candidate['family']}]")
        print("==============================")

        result = evaluate_candidate(candidate, X_train, y_train, X_test, y_test, cv)
        all_results.append(result)

        print(f"MAE           : {result['holdout_mae']:.4f}")
        print(f"RMSE          : {result['holdout_rmse']:.4f}")
        print(f"R2            : {result['holdout_r2']:.4f}")

    leaderboard = pd.DataFrame([
        {
            "name": r["name"],
            "family": r["family"],
            "cv_best_mae": r["cv_best_mae"],
            "holdout_mae": r["holdout_mae"],
            "holdout_rmse": r["holdout_rmse"],
            "holdout_r2": r["holdout_r2"],
        }
        for r in all_results
    ]).sort_values(
        by=["holdout_mae", "holdout_rmse"],
        ascending=[True, True],
    ).reset_index(drop=True)

    winner_name = leaderboard.iloc[0]["name"]
    winner = next(r for r in all_results if r["name"] == winner_name)

    print("\n==============================")
    print("FINAL LEADERBOARD")
    print("==============================")
    print(leaderboard[["name", "family", "holdout_mae", "holdout_rmse", "holdout_r2"]])

    winner_print = {
        "name": winner["name"],
        "family": winner["family"],
        "best_params": winner["best_params"],
        "cv_best_mae": winner["cv_best_mae"],
        "holdout_mae": winner["holdout_mae"],
        "holdout_rmse": winner["holdout_rmse"],
        "holdout_r2": winner["holdout_r2"],
    }

    print("\nWinner:")
    print(json.dumps(winner_print, ensure_ascii=False, indent=2))

    save_artifacts(
        out_dir=out_dir,
        winner=winner,
        leaderboard=leaderboard,
        rows_used=len(df),
        feature_columns=FEATURE_COLUMNS,
    )

    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
