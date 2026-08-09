import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from preprocessing import (
    CANONICAL_COLUMNS,
    harmonize_schema,
    prepare_features,
    ensure_required_columns,
)


MIN_FOUNDED_YEAR = 1850
MAX_FOUNDED_YEAR = 2025


REQUIRED_COLUMNS = [
    "name",
    "description",
    "price_category",
    "country_origin",
    "domain",
    "presence_world",
    "presence_russia",
    "presence_regions",
    "plans",
    "founded",
]


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def build_schema_report(df_raw: pd.DataFrame, df: pd.DataFrame) -> Dict:
    raw_cols = list(df_raw.columns)
    prepared_cols = list(df.columns)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    missing_canonical = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    extra_columns = [c for c in prepared_cols if c not in CANONICAL_COLUMNS]

    return {
        "raw_columns": raw_cols,
        "prepared_columns": prepared_cols,
        "missing_required_columns": missing_required,
        "missing_canonical_columns": missing_canonical,
        "extra_columns_after_preparation": extra_columns,
        "raw_n_columns": len(raw_cols),
        "prepared_n_columns": len(prepared_cols),
    }


def build_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "missing_count": n_missing,
            "missing_rate": float(n_missing / n) if n else 0.0,
            "dtype": str(df[col].dtype),
        })

    return pd.DataFrame(rows).sort_values(
        by=["missing_rate", "missing_count"],
        ascending=[False, False],
    )


def build_text_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [
        c for c in [
            "name",
            "name_clean",
            "description",
            "country_origin",
            "domain",
            "price_category",
        ]
        if c in df.columns
    ]

    rows = []
    for col in text_cols:
        s = df[col].fillna("").astype(str)

        rows.append({
            "column": col,
            "empty_count": int((s.str.strip() == "").sum()),
            "empty_rate": float((s.str.strip() == "").mean()) if len(s) else 0.0,
            "avg_length": float(s.str.len().mean()) if len(s) else 0.0,
            "median_length": float(s.str.len().median()) if len(s) else 0.0,
            "n_unique": int(s.nunique()),
        })

    return pd.DataFrame(rows).sort_values(by="empty_rate", ascending=False)


def build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c for c in [
            "founded",
            "presence_world",
            "plans",
            "total_rented_area",
            "presence_own",
            "presence_franchise",
            "n_regions",
            "desc_len",
            "desc_has_digits",
            "desc_year_mentions",
            "has_total_rented_area",
        ]
        if c in df.columns
    ]

    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()

        if len(non_null) == 0:
            rows.append({
                "column": col,
                "count_non_null": 0,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
                "mean": None,
            })
            continue

        rows.append({
            "column": col,
            "count_non_null": int(non_null.shape[0]),
            "min": safe_float(non_null.min()),
            "p25": safe_float(non_null.quantile(0.25)),
            "median": safe_float(non_null.median()),
            "p75": safe_float(non_null.quantile(0.75)),
            "max": safe_float(non_null.max()),
            "mean": safe_float(non_null.mean()),
        })

    return pd.DataFrame(rows)


def build_anomaly_report(df: pd.DataFrame) -> Dict:
    report = {}

    if "founded" in df.columns:
        founded = pd.to_numeric(df["founded"], errors="coerce")
        report["founded_lt_1850"] = int((founded < MIN_FOUNDED_YEAR).sum())
        report["founded_gt_2025"] = int((founded > MAX_FOUNDED_YEAR).sum())
        report["founded_missing"] = int(founded.isna().sum())

    if "presence_world" in df.columns:
        s = pd.to_numeric(df["presence_world"], errors="coerce")
        report["presence_world_negative"] = int((s < 0).sum())

    if "plans" in df.columns:
        s = pd.to_numeric(df["plans"], errors="coerce")
        report["plans_negative"] = int((s < 0).sum())

    if "total_rented_area" in df.columns:
        s = pd.to_numeric(df["total_rented_area"], errors="coerce")
        report["total_rented_area_negative"] = int((s < 0).sum())
        report["total_rented_area_known_count"] = int(s.notna().sum())
        report["total_rented_area_known_rate"] = float(s.notna().mean()) if len(df) else 0.0

    if "description" in df.columns:
        desc = df["description"].fillna("").astype(str)
        report["description_empty"] = int((desc.str.strip() == "").sum())

    return report


def build_duplicate_report(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}

    if "name" in df.columns:
        exact_name_dupes = (
            df.groupby("name")
            .size()
            .reset_index(name="rows")
            .sort_values(by="rows", ascending=False)
        )
        out["exact_name_duplicates"] = exact_name_dupes[exact_name_dupes["rows"] > 1].copy()

    if "name_clean" in df.columns:
        clean_name_dupes = (
            df.groupby("name_clean")
            .size()
            .reset_index(name="rows")
            .sort_values(by="rows", ascending=False)
        )
        out["clean_name_duplicates"] = clean_name_dupes[clean_name_dupes["rows"] > 1].copy()

        if "name" in df.columns:
            ambiguous = (
                df.groupby("name_clean")["name"]
                .nunique()
                .reset_index(name="n_original_name_variants")
                .sort_values(by="n_original_name_variants", ascending=False)
            )
            out["ambiguous_clean_names"] = ambiguous[ambiguous["n_original_name_variants"] > 1].copy()

    return out


def build_target_reports(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    reports = {}

    if "domain" in df.columns:
        domain_counts = (
            df["domain"]
            .fillna("MISSING")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("domain")
            .reset_index(name="count")
        )
        domain_counts["rate"] = domain_counts["count"] / len(df) if len(df) else 0.0
        reports["domain_distribution"] = domain_counts

    if "price_category" in df.columns:
        price_counts = (
            df["price_category"]
            .fillna("MISSING")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("price_category")
            .reset_index(name="count")
        )
        price_counts["rate"] = price_counts["count"] / len(df) if len(df) else 0.0
        reports["price_category_distribution"] = price_counts

        token_rows = []
        for value in df["price_category"].fillna("неизвестно").astype(str):
            parts = [p.strip() for p in value.split(";") if p.strip()]
            token_rows.extend(parts)

        if token_rows:
            token_counts = (
                pd.Series(token_rows)
                .value_counts()
                .rename_axis("label")
                .reset_index(name="count")
            )
            token_counts["rate"] = token_counts["count"] / len(df) if len(df) else 0.0
        else:
            token_counts = pd.DataFrame(columns=["label", "count", "rate"])

        reports["price_category_label_distribution"] = token_counts

    if "founded" in df.columns:
        founded = pd.to_numeric(df["founded"], errors="coerce")
        decade = (np.floor(founded / 10) * 10).astype("Int64")
        decade_counts = (
            decade.dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis("decade")
            .reset_index(name="count")
        )
        decade_counts["rate"] = decade_counts["count"] / max(1, founded.notna().sum())
        reports["founded_decade_distribution"] = decade_counts

    return reports


def build_conflict_report(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    reports = {}

    key_cols = [c for c in ["name", "country_origin", "domain", "price_category"] if c in df.columns]

    if len(key_cols) == 4 and "founded" in df.columns:
        founded_conflicts = (
            df.groupby(key_cols)["founded"]
            .nunique(dropna=True)
            .reset_index(name="n_founded_variants")
        )
        founded_conflicts = founded_conflicts[founded_conflicts["n_founded_variants"] > 1].copy()
        reports["founded_conflicts"] = founded_conflicts.sort_values(
            by="n_founded_variants", ascending=False
        )

    if "name_clean" in df.columns and "domain" in df.columns:
        domain_conflicts = (
            df.groupby("name_clean")["domain"]
            .nunique(dropna=True)
            .reset_index(name="n_domain_variants")
        )
        domain_conflicts = domain_conflicts[domain_conflicts["n_domain_variants"] > 1].copy()
        reports["name_clean_domain_conflicts"] = domain_conflicts.sort_values(
            by="n_domain_variants", ascending=False
        )

    if "name_clean" in df.columns and "country_origin" in df.columns:
        country_conflicts = (
            df.groupby("name_clean")["country_origin"]
            .nunique(dropna=True)
            .reset_index(name="n_country_variants")
        )
        country_conflicts = country_conflicts[country_conflicts["n_country_variants"] > 1].copy()
        reports["name_clean_country_conflicts"] = country_conflicts.sort_values(
            by="n_country_variants", ascending=False
        )

    return reports


def build_training_readiness_report(df: pd.DataFrame) -> Dict:
    report = {"rows_total": int(len(df))}

    if "domain" in df.columns:
        tmp = df[df["domain"].notna()].copy()
        tmp["domain"] = tmp["domain"].astype(str).str.strip()
        value_counts = tmp["domain"].value_counts()
        keep = value_counts[value_counts >= 5]
        report["domain_task"] = {
            "rows_non_null_target": int(tmp.shape[0]),
            "n_classes_all": int(value_counts.shape[0]),
            "n_classes_ge_5": int(keep.shape[0]),
            "rows_after_drop_rare_classes": int(tmp[tmp["domain"].isin(keep.index)].shape[0]),
        }

    if "founded" in df.columns:
        founded = pd.to_numeric(df["founded"], errors="coerce")
        valid = founded.between(MIN_FOUNDED_YEAR, MAX_FOUNDED_YEAR)
        report["founded_task"] = {
            "rows_non_null_target": int(founded.notna().sum()),
            "rows_in_valid_range_1850_2025": int(valid.sum()),
        }

    if "price_category" in df.columns:
        non_null = df["price_category"].fillna("неизвестно").astype(str)
        report["price_category_task"] = {
            "rows_non_null_target": int((non_null != "").sum()),
            "n_unique_combinations": int(non_null.nunique()),
        }

    return report


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="QA checks for retail enrichment project.")
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df_raw = pd.read_csv(args.csv_path)

    print("Harmonizing schema...")
    df_harmonized = harmonize_schema(df_raw)

    print("Running feature preparation...")
    df = prepare_features(df_harmonized)

    print("Checking required columns...")
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    schema_report = build_schema_report(df_raw, df)
    missing_report = build_missing_report(df)
    text_quality_report = build_text_quality_report(df)
    numeric_summary = build_numeric_summary(df)
    anomaly_report = build_anomaly_report(df)
    duplicate_reports = build_duplicate_report(df)
    target_reports = build_target_reports(df)
    conflict_reports = build_conflict_report(df)
    training_readiness = build_training_readiness_report(df)

    summary = {
        "dataset_shape_raw": [int(df_raw.shape[0]), int(df_raw.shape[1])],
        "dataset_shape_prepared": [int(df.shape[0]), int(df.shape[1])],
        "schema_report": schema_report,
        "anomaly_report": anomaly_report,
        "training_readiness": training_readiness,
    }

    if missing_required:
        summary["required_columns_check"] = {
            "status": "FAILED",
            "missing_required_columns": missing_required,
        }
    else:
        summary["required_columns_check"] = {
            "status": "OK",
            "missing_required_columns": [],
        }

    print("Saving reports...")
    (out_dir / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_dataframe(missing_report, out_dir / "missing_report.csv")
    save_dataframe(text_quality_report, out_dir / "text_quality_report.csv")
    save_dataframe(numeric_summary, out_dir / "numeric_summary.csv")

    for name, rep in duplicate_reports.items():
        save_dataframe(rep, out_dir / f"{name}.csv")

    for name, rep in target_reports.items():
        save_dataframe(rep, out_dir / f"{name}.csv")

    for name, rep in conflict_reports.items():
        save_dataframe(rep, out_dir / f"{name}.csv")

    preview_cols = [c for c in [
        "name",
        "name_clean",
        "country_origin",
        "domain",
        "price_category",
        "founded",
        "presence_world",
        "presence_russia",
        "presence_own",
        "presence_franchise",
        "n_regions",
        "desc_len",
        "desc_has_digits",
        "desc_year_mentions",
        "has_total_rented_area",
    ] if c in df.columns]

    save_dataframe(df[preview_cols].head(1000), out_dir / "prepared_preview_1000.csv")

    print("\n==============================")
    print("QA CHECKS COMPLETE")
    print("==============================")
    print(f"Rows raw              : {df_raw.shape[0]}")
    print(f"Cols raw              : {df_raw.shape[1]}")
    print(f"Rows prepared         : {df.shape[0]}")
    print(f"Cols prepared         : {df.shape[1]}")
    print(f"Required columns OK   : {not bool(missing_required)}")

    if "domain_task" in training_readiness:
        print("\n[domain]")
        print(f"Rows with target      : {training_readiness['domain_task']['rows_non_null_target']}")
        print(f"Classes total         : {training_readiness['domain_task']['n_classes_all']}")
        print(f"Classes >= 5 rows     : {training_readiness['domain_task']['n_classes_ge_5']}")
        print(f"Rows after class cut  : {training_readiness['domain_task']['rows_after_drop_rare_classes']}")

    if "founded_task" in training_readiness:
        print("\n[founded]")
        print(f"Rows with target      : {training_readiness['founded_task']['rows_non_null_target']}")
        print(f"Rows in valid range   : {training_readiness['founded_task']['rows_in_valid_range_1850_2025']}")

    if "price_category_task" in training_readiness:
        print("\n[price_category]")
        print(f"Rows with target      : {training_readiness['price_category_task']['rows_non_null_target']}")
        print(f"Unique combinations   : {training_readiness['price_category_task']['n_unique_combinations']}")

    if "total_rented_area_known_rate" in anomaly_report:
        print("\n[total_rented_area]")
        print(f"Known count           : {anomaly_report['total_rented_area_known_count']}")
        print(f"Known rate            : {anomaly_report['total_rented_area_known_rate']:.4f}")

    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
