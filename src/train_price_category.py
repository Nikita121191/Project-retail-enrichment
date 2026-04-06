import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

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
    "country_origin",
    "domain",
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
    "domain",
]


PRICE_LABEL_ORDER = [
    "дисконт",
    "ниже среднего",
    "средний",
    "выше среднего",
    "люкс / премиум",
    "неизвестно",
]


class MultiLabelMostFrequentDummy(BaseEstimator):
    def fit(self, X, y):
        self.label_frequencies_ = np.mean(y, axis=0)
        self.most_frequent_label_idx_ = int(np.argmax(self.label_frequencies_))
        self.n_labels_ = y.shape[1]
        return self

    def predict(self, X):
        n_rows = X.shape[0]
        out = np.zeros((n_rows, self.n_labels_), dtype=int)
        out[:, self.most_frequent_label_idx_] = 1
        return out

    def predict_proba(self, X):
        n_rows = X.shape[0]
        probs = np.tile(self.label_frequencies_, (n_rows, 1))
        return probs


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def normalize_price_string(s: object) -> str:
    if pd.isna(s):
        return "неизвестно"

    s = str(s).strip().lower()
    s = s.replace("\u200b", "")
    s = s.replace("\xa0", " ")
    s = s.replace(",", ";")
    s = s.replace(":", ";")

    parts = [p.strip() for p in s.split(";") if p.strip()]
    parts = list(dict.fromkeys(parts))

    return ";".join(parts) if parts else "неизвестно"


def split_price_labels(s: str):
    parts = [p.strip() for p in str(s).split(";") if p.strip()]
    return parts if parts else ["неизвестно"]


def make_multilabel_target(series: pd.Series):
    normalized = series.apply(normalize_price_string)
    label_lists = normalized.apply(split_price_labels)

    mlb = MultiLabelBinarizer(classes=PRICE_LABEL_ORDER)
    y = mlb.fit_transform(label_lists)

    return normalized, label_lists, mlb, y


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

    df["description"] = df["description"].fillna("").astype(str)
    df["name_clean"] = df["name_clean"].fillna("").astype(str)
    df["country_origin"] = df["country_origin"].fillna("Неизвестно").astype(str)
    df["domain"] = df["domain"].fillna("Неизвестно").astype(str)
    df["price_category"] = df["price_category"].fillna("неизвестно").astype(str)

    return df


def build_candidates():
    preprocessor = build_preprocessor()

    candidates = [
        {
            "name": "DummyMostFrequent",
            "family": "baseline",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", MultiLabelMostFrequentDummy()),
            ]),
            "param_grid": None,
        },
        {
            "name": "LogisticRegression",
            "family": "linear",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    )
                )),
            ]),
            "param_grid": {
                "model__estimator__C": [0.3, 1.0, 3.0],
            },
        },
        {
            "name": "RandomForest",
            "family": "bagging",
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("model", OneVsRestClassifier(
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )
                )),
            ]),
            "param_grid": {
                "model__estimator__max_depth": [12, 20],
                "model__estimator__min_samples_leaf": [1, 2],
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
                    ("model", OneVsRestClassifier(
                        XGBClassifier(
                            n_estimators=300,
                            learning_rate=0.1,
                            max_depth=4,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            eval_metric="logloss",
                        )
                    )),
                ]),
                "param_grid": {
                    "model__estimator__learning_rate": [0.05, 0.1],
                    "model__estimator__max_depth": [4, 6],
                },
            }
        )

    return candidates


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray, labels: list) -> dict:
    thresholds = {}

    for i, label in enumerate(labels):
        best_thr = 0.5
        best_f1 = -1.0

        for thr in np.arange(0.10, 0.91, 0.05):
            pred = (y_prob[:, i] >= thr).astype(int)
            score = f1_score(y_true[:, i], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = float(np.round(thr, 2))

        thresholds[label] = best_thr

    return thresholds


def apply_thresholds(y_prob: np.ndarray, thresholds: dict, labels: list) -> np.ndarray:
    out = np.zeros_like(y_prob, dtype=int)

    for i, label in enumerate(labels):
        thr = thresholds[label]
        out[:, i] = (y_prob[:, i] >= thr).astype(int)

    row_sums = out.sum(axis=1)
    empty_rows = np.where(row_sums == 0)[0]

    if len(empty_rows) > 0:
        max_idx = np.argmax(y_prob[empty_rows], axis=1)
        out[empty_rows, max_idx] = 1

    return out


def multilabel_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
    )


def get_probabilities(fitted_model, X) -> np.ndarray:
    """
    ВАЖНО:
    predict_proba вызываем у всего pipeline, а не у внутренней модели,
    чтобы сначала отработал preprocessor.
    """
    if hasattr(fitted_model, "predict_proba"):
        probs = fitted_model.predict_proba(X)

        if isinstance(probs, list):
            cols = []
            for p in probs:
                p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] == 2:
                    cols.append(p[:, 1])
                else:
                    cols.append(p.reshape(-1))
            return np.column_stack(cols)

        if isinstance(probs, np.ndarray):
            if probs.ndim == 3 and probs.shape[2] == 2:
                return probs[:, :, 1]
            return probs

    preds = fitted_model.predict(X)
    return np.asarray(preds, dtype=float)


def cross_validated_threshold_score(candidate, X_train, y_train, labels, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train), start=1):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        model = clone(candidate["pipeline"])
        model.fit(X_tr, y_tr)

        y_prob = get_probabilities(model, X_va)
        thresholds = optimize_thresholds(y_va, y_prob, labels)
        y_pred_thr = apply_thresholds(y_prob, thresholds, labels)

        score = f1_score(y_va, y_pred_thr, average="macro", zero_division=0)
        fold_scores.append(float(score))

        print(f"CV fold {fold_idx}: thresholded macro F1 = {score:.4f}")

    return float(np.mean(fold_scores))


def evaluate_candidate(candidate, X_train, y_train, X_test, y_test, labels, inner_cv):
    pipe = clone(candidate["pipeline"])

    if candidate["param_grid"]:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=candidate["param_grid"],
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted_model = search.best_estimator_
        best_params = search.best_params_
        cv_best_macro_f1 = float(search.best_score_)
    else:
        fitted_model = pipe.fit(X_train, y_train)
        best_params = {}
        cv_best_macro_f1 = None

    cv_threshold_macro_f1 = cross_validated_threshold_score(
        candidate=candidate,
        X_train=X_train,
        y_train=y_train,
        labels=labels,
        n_splits=5,
    )

    y_pred_direct = fitted_model.predict(X_test)
    direct_metrics = multilabel_metrics_dict(y_test, y_pred_direct)

    y_prob_train = get_probabilities(fitted_model, X_train)
    thresholds = optimize_thresholds(y_train, y_prob_train, labels)

    y_prob_test = get_probabilities(fitted_model, X_test)
    y_pred_threshold = apply_thresholds(y_prob_test, thresholds, labels)
    threshold_metrics = multilabel_metrics_dict(y_test, y_pred_threshold)

    report_text = classification_report_text(y_test, y_pred_threshold, labels)

    print("\n--- DIRECT PREDICT ---")
    print(f"Macro F1      : {direct_metrics['macro_f1']:.4f}")
    print(f"Micro F1      : {direct_metrics['micro_f1']:.4f}")
    print(f"Weighted F1   : {direct_metrics['weighted_f1']:.4f}")
    print(f"Samples F1    : {direct_metrics['samples_f1']:.4f}")

    print("\n--- THRESHOLDED PREDICT ---")
    print(f"Macro F1      : {threshold_metrics['macro_f1']:.4f}")
    print(f"Micro F1      : {threshold_metrics['micro_f1']:.4f}")
    print(f"Weighted F1   : {threshold_metrics['weighted_f1']:.4f}")
    print(f"Samples F1    : {threshold_metrics['samples_f1']:.4f}")

    print("\n--- CV DIAGNOSTICS ---")
    print(f"GridSearch CV macro F1        : {cv_best_macro_f1}")
    print(f"Manual threshold-CV macro F1  : {cv_threshold_macro_f1:.4f}")

    print("\nUsed thresholds:")
    for label in labels:
        print(f"  {label}: {thresholds[label]:.2f}")

    print("\nClassification report (thresholded predict):")
    print(report_text)

    return {
        "name": candidate["name"],
        "family": candidate["family"],
        "model": fitted_model,
        "best_params": best_params,
        "cv_best_macro_f1": cv_best_macro_f1,
        "cv_threshold_macro_f1": cv_threshold_macro_f1,
        "direct_macro_f1": direct_metrics["macro_f1"],
        "direct_micro_f1": direct_metrics["micro_f1"],
        "threshold_macro_f1": threshold_metrics["macro_f1"],
        "threshold_micro_f1": threshold_metrics["micro_f1"],
        "threshold_weighted_f1": threshold_metrics["weighted_f1"],
        "threshold_samples_f1": threshold_metrics["samples_f1"],
        "thresholds": thresholds,
        "classification_report": report_text,
    }


def save_artifacts(
    out_dir: Path,
    winner: dict,
    leaderboard: pd.DataFrame,
    labels: list,
    rows_used: int,
    feature_columns: list,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    joblib.dump(winner["model"], out_dir / "price_category_model.joblib")

    with open(out_dir / "price_category_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(winner["thresholds"], f, ensure_ascii=False, indent=2)

    with open(out_dir / "price_category_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    report = {
        "target": "price_category",
        "winner_name": winner["name"],
        "winner_family": winner["family"],
        "rows_used": int(rows_used),
        "feature_columns": feature_columns,
        "labels": labels,
        "best_params": winner["best_params"],
        "cv_best_macro_f1": winner["cv_best_macro_f1"],
        "cv_threshold_macro_f1": winner["cv_threshold_macro_f1"],
        "direct_macro_f1": winner["direct_macro_f1"],
        "direct_micro_f1": winner["direct_micro_f1"],
        "threshold_macro_f1": winner["threshold_macro_f1"],
        "threshold_micro_f1": winner["threshold_micro_f1"],
        "threshold_weighted_f1": winner["threshold_weighted_f1"],
        "threshold_samples_f1": winner["threshold_samples_f1"],
    }

    with open(out_dir / "price_category_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "price_category_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(winner["classification_report"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/kaggle/input/datasets/nikitasadovoy/russian-retail/russian_retail.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/kaggle/working/artifacts/price_category",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )
    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    df = load_and_prepare_data(args.csv_path)

    _, _, mlb, y = make_multilabel_target(df["price_category"])
    labels = list(mlb.classes_)

    X = df[FEATURE_COLUMNS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()

    all_results = []

    for candidate in candidates:
        print("\n==============================")
        print(f"Training: {candidate['name']} [{candidate['family']}]")
        print("==============================")

        result = evaluate_candidate(
            candidate=candidate,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            labels=labels,
            inner_cv=inner_cv,
        )
        all_results.append(result)

    leaderboard = pd.DataFrame([
        {
            "name": r["name"],
            "family": r["family"],
            "cv_best_macro_f1": r["cv_best_macro_f1"],
            "cv_threshold_macro_f1": r["cv_threshold_macro_f1"],
            "direct_macro_f1": r["direct_macro_f1"],
            "threshold_macro_f1": r["threshold_macro_f1"],
            "threshold_micro_f1": r["threshold_micro_f1"],
        }
        for r in all_results
    ]).sort_values(
        by=["threshold_macro_f1", "threshold_micro_f1"],
        ascending=[False, False],
    ).reset_index(drop=True)

    winner_name = leaderboard.iloc[0]["name"]
    winner = next(r for r in all_results if r["name"] == winner_name)

    print("\n==============================")
    print("FINAL LEADERBOARD")
    print("==============================")
    print(leaderboard)

    winner_print = {
        "name": winner["name"],
        "family": winner["family"],
        "best_params": winner["best_params"],
        "cv_best_macro_f1": winner["cv_best_macro_f1"],
        "cv_threshold_macro_f1": winner["cv_threshold_macro_f1"],
        "direct_macro_f1": winner["direct_macro_f1"],
        "direct_micro_f1": winner["direct_micro_f1"],
        "threshold_macro_f1": winner["threshold_macro_f1"],
        "threshold_micro_f1": winner["threshold_micro_f1"],
        "threshold_weighted_f1": winner["threshold_weighted_f1"],
        "threshold_samples_f1": winner["threshold_samples_f1"],
    }

    print("\nWinner:")
    print(json.dumps(winner_print, ensure_ascii=False, indent=2))

    save_artifacts(
        out_dir=out_dir,
        winner=winner,
        leaderboard=leaderboard,
        labels=labels,
        rows_used=len(df),
        feature_columns=FEATURE_COLUMNS,
    )

    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
