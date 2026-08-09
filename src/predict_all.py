import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import prepare_features


DOMAIN_FEATURE_COLUMNS = [
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

FOUNDED_FEATURE_COLUMNS = [
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

PRICE_FEATURE_COLUMNS = [
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

TEXT_COLUMNS = [
    "description",
    "name_clean",
]

CATEGORICAL_COLUMNS = [
    "country_origin",
    "domain",
    "price_category",
]

NUMERIC_COLUMNS = [
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


def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            if col in TEXT_COLUMNS:
                df[col] = ""
            elif col in CATEGORICAL_COLUMNS:
                if col == "country_origin":
                    df[col] = "Неизвестно"
                elif col == "price_category":
                    df[col] = "неизвестно"
                else:
                    df[col] = "Неизвестно"
            else:
                df[col] = np.nan

    return df


def sanitize_for_model(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    df = ensure_columns(df, feature_columns).copy()

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            if col == "country_origin":
                df[col] = df[col].fillna("Неизвестно").astype(str)
            elif col == "price_category":
                df[col] = df[col].fillna("неизвестно").astype(str)
            else:
                df[col] = df[col].fillna("Неизвестно").astype(str)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[feature_columns].copy()


def get_price_probabilities(fitted_model, X) -> np.ndarray:
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

def run_pipeline(df_raw: pd.DataFrame, artifacts_dir: str | Path) -> pd.DataFrame:
    artifacts_dir = Path(artifacts_dir)

    df = prepare_features(df_raw)
    result = df_raw.copy()

    # ---------- DOMAIN ----------
    domain_model_path = artifacts_dir / "domain" / "domain_model.joblib"
    domain_encoder_path = artifacts_dir / "domain" / "domain_label_encoder.joblib"

    if domain_model_path.exists():
        domain_model = joblib.load(domain_model_path)

        X_domain = sanitize_for_model(df, DOMAIN_FEATURE_COLUMNS)
        domain_pred = domain_model.predict(X_domain)

        if np.issubdtype(np.asarray(domain_pred).dtype, np.integer) and domain_encoder_path.exists():
            domain_encoder = joblib.load(domain_encoder_path)
            domain_pred = domain_encoder.inverse_transform(domain_pred)

        result["pred_domain"] = domain_pred
    else:
        result["pred_domain"] = np.nan

    # ---------- FOUNDED ----------
    founded_model_path = artifacts_dir / "founded" / "founded_model.joblib"

    if founded_model_path.exists():
        founded_model = joblib.load(founded_model_path)

        founded_df = df.copy()
        if "pred_domain" in result.columns:
            founded_df["domain"] = result["pred_domain"].fillna(founded_df.get("domain", "Неизвестно"))

        X_founded = sanitize_for_model(founded_df, FOUNDED_FEATURE_COLUMNS)
        founded_pred = founded_model.predict(X_founded)
        founded_pred = np.clip(founded_pred, 0, None)

        result["pred_founded"] = founded_pred
        result["pred_founded_rounded"] = np.round(founded_pred).astype(int)
    else:
        result["pred_founded"] = np.nan
        result["pred_founded_rounded"] = np.nan

    # ---------- PRICE ----------
    price_model_path = artifacts_dir / "price_category" / "price_category_model.joblib"
    price_thresholds_path = artifacts_dir / "price_category" / "price_category_thresholds.json"
    price_labels_path = artifacts_dir / "price_category" / "price_category_labels.json"

    if price_model_path.exists() and price_thresholds_path.exists() and price_labels_path.exists():
        price_model = joblib.load(price_model_path)

        with open(price_thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

        with open(price_labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        price_df = df.copy()

        if "pred_domain" in result.columns:
            price_df["domain"] = result["pred_domain"].fillna(price_df.get("domain", "Неизвестно"))

        if "pred_founded" in result.columns:
            current_founded = price_df.get("founded", pd.Series(np.nan, index=price_df.index))
            price_df["founded"] = pd.Series(result["pred_founded"], index=price_df.index).fillna(current_founded)

        X_price = sanitize_for_model(price_df, PRICE_FEATURE_COLUMNS)

        y_prob = get_price_probabilities(price_model, X_price)
        y_pred_bin = apply_thresholds(y_prob, thresholds, labels)
        y_pred_text = multilabel_rows_to_strings(y_pred_bin, labels)

        result["pred_price_category"] = y_pred_text

    else:
        result["pred_price_category"] = np.nan

    return result
def apply_thresholds(y_prob: np.ndarray, thresholds: dict, labels: list) -> np.ndarray:
    out = np.zeros_like(y_prob, dtype=int)

    for i, label in enumerate(labels):
        thr = float(thresholds[label])
        out[:, i] = (y_prob[:, i] >= thr).astype(int)

    row_sums = out.sum(axis=1)
    empty_rows = np.where(row_sums == 0)[0]

    if len(empty_rows) > 0:
        max_idx = np.argmax(y_prob[empty_rows], axis=1)
        out[empty_rows, max_idx] = 1

    return out


def multilabel_rows_to_strings(y_bin: np.ndarray, labels: list) -> list:
    results = []
    for row in y_bin:
        active = [label for label, val in zip(labels, row) if val == 1]
        results.append("; ".join(active) if active else "неизвестно")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--artifacts_dir", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")

    if not args.artifacts_dir.is_dir():
        parser.error(f"Artifacts directory not found: {args.artifacts_dir}")

    artifacts_dir = args.artifacts_dir
    out_path = args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df_raw = pd.read_csv(args.csv_path)

    print("Preparing features...")
    df = prepare_features(df_raw)

    result = df_raw.copy()

    # ---------- DOMAIN ----------
    domain_model_path = artifacts_dir / "domain" / "domain_model.joblib"
    domain_encoder_path = artifacts_dir / "domain" / "domain_label_encoder.joblib"

    if domain_model_path.exists():
        print("Loading domain model...")
        domain_model = joblib.load(domain_model_path)

        X_domain = sanitize_for_model(df, DOMAIN_FEATURE_COLUMNS)
        domain_pred = domain_model.predict(X_domain)

        if np.issubdtype(np.asarray(domain_pred).dtype, np.integer) and domain_encoder_path.exists():
            domain_encoder = joblib.load(domain_encoder_path)
            domain_pred = domain_encoder.inverse_transform(domain_pred)

        result["pred_domain"] = domain_pred
    else:
        print("Domain model not found, skipping.")
        result["pred_domain"] = np.nan

    # ---------- FOUNDED ----------
    founded_model_path = artifacts_dir / "founded" / "founded_model.joblib"

    if founded_model_path.exists():
        print("Loading founded model...")
        founded_model = joblib.load(founded_model_path)

        founded_df = df.copy()
        if "pred_domain" in result.columns:
            founded_df["domain"] = result["pred_domain"].fillna(founded_df.get("domain", "Неизвестно"))

        X_founded = sanitize_for_model(founded_df, FOUNDED_FEATURE_COLUMNS)
        founded_pred = founded_model.predict(X_founded)
        founded_pred = np.clip(founded_pred, 0, None)

        result["pred_founded"] = founded_pred
        result["pred_founded_rounded"] = np.round(founded_pred).astype(int)
    else:
        print("Founded model not found, skipping.")
        result["pred_founded"] = np.nan
        result["pred_founded_rounded"] = np.nan

    # ---------- PRICE CATEGORY ----------
    price_model_path = artifacts_dir / "price_category" / "price_category_model.joblib"
    price_thresholds_path = artifacts_dir / "price_category" / "price_category_thresholds.json"
    price_labels_path = artifacts_dir / "price_category" / "price_category_labels.json"

    if price_model_path.exists() and price_thresholds_path.exists() and price_labels_path.exists():
        print("Loading price_category model...")
        price_model = joblib.load(price_model_path)

        with open(price_thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

        with open(price_labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        price_df = df.copy()

        if "pred_domain" in result.columns:
            price_df["domain"] = result["pred_domain"].fillna(price_df.get("domain", "Неизвестно"))

        if "pred_founded" in result.columns:
            current_founded = price_df.get("founded", pd.Series(np.nan, index=price_df.index))
            price_df["founded"] = pd.Series(result["pred_founded"], index=price_df.index).fillna(current_founded)

        X_price = sanitize_for_model(price_df, PRICE_FEATURE_COLUMNS)

        y_prob = get_price_probabilities(price_model, X_price)
        y_pred_bin = apply_thresholds(y_prob, thresholds, labels)
        y_pred_text = multilabel_rows_to_strings(y_pred_bin, labels)

        result["pred_price_category"] = y_pred_text

        for i, label in enumerate(labels):
            safe_label = (
                label.replace(" / ", "_")
                .replace(" ", "_")
                .replace("-", "_")
            )
            result[f"pred_price_prob__{safe_label}"] = y_prob[:, i]
            result[f"pred_price_label__{safe_label}"] = y_pred_bin[:, i]
    else:
        print("Price category artifacts not found, skipping.")
        result["pred_price_category"] = np.nan

    print("Saving predictions...")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows_scored": int(len(result)),
        "domain_model_used": bool(domain_model_path.exists()),
        "founded_model_used": bool(founded_model_path.exists()),
        "price_category_model_used": bool(price_model_path.exists()),
        "output_path": str(out_path),
    }

    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("PREDICT ALL COMPLETE")
    print("==============================")
    print(f"Rows scored      : {len(result)}")
    print(f"Saved to         : {out_path}")


if __name__ == "__main__":
    main()
