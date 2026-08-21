from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_contract import (
    KNOWN_PRICE_LABELS,
    MAX_FOUNDED_YEAR,
    MIN_FOUNDED_YEAR,
    UNKNOWN_DOMAIN,
    UNKNOWN_PRICE_LABEL,
    split_price_target,
)
from preprocessing import (
    CANONICAL_COLUMNS,
    harmonize_schema,
    prepare_features,
)


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


def safe_float(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def build_schema_report(
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
) -> dict:
    raw_columns = list(
        df_raw.columns
    )
    prepared_columns = list(
        df.columns
    )

    missing_required = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    missing_canonical = [
        column
        for column in CANONICAL_COLUMNS
        if column not in df.columns
    ]

    extra_columns = [
        column
        for column in prepared_columns
        if column not in CANONICAL_COLUMNS
    ]

    return {
        "raw_columns": raw_columns,
        "prepared_columns": (
            prepared_columns
        ),
        "missing_required_columns": (
            missing_required
        ),
        "missing_canonical_columns": (
            missing_canonical
        ),
        "extra_columns_after_preparation": (
            extra_columns
        ),
        "raw_n_columns": len(
            raw_columns
        ),
        "prepared_n_columns": len(
            prepared_columns
        ),
    }


def build_missing_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    n_rows = len(df)

    for column in df.columns:
        missing_count = int(
            df[column].isna().sum()
        )

        rows.append({
            "column": column,
            "missing_count": (
                missing_count
            ),
            "missing_rate": (
                float(
                    missing_count
                    / n_rows
                )
                if n_rows
                else 0.0
            ),
            "dtype": str(
                df[column].dtype
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "missing_rate",
                "missing_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )


def build_text_quality_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    text_columns = [
        column
        for column in [
            "name",
            "name_clean",
            "description",
            "country_origin",
            "domain",
            "price_category",
        ]
        if column in df.columns
    ]

    rows = []

    for column in text_columns:
        series = (
            df[column]
            .fillna("")
            .astype(str)
        )

        empty_mask = (
            series.str.strip() == ""
        )

        rows.append({
            "column": column,
            "empty_count": int(
                empty_mask.sum()
            ),
            "empty_rate": (
                float(
                    empty_mask.mean()
                )
                if len(series)
                else 0.0
            ),
            "avg_length": (
                float(
                    series.str.len().mean()
                )
                if len(series)
                else 0.0
            ),
            "median_length": (
                float(
                    series.str.len().median()
                )
                if len(series)
                else 0.0
            ),
            "n_unique": int(
                series.nunique()
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            by="empty_rate",
            ascending=False,
        )
    )


def build_numeric_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in [
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
        if column in df.columns
    ]

    rows = []

    for column in numeric_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )
        non_null = (
            series.dropna()
        )

        if len(non_null) == 0:
            rows.append({
                "column": column,
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
            "column": column,
            "count_non_null": int(
                len(non_null)
            ),
            "min": safe_float(
                non_null.min()
            ),
            "p25": safe_float(
                non_null.quantile(
                    0.25
                )
            ),
            "median": safe_float(
                non_null.median()
            ),
            "p75": safe_float(
                non_null.quantile(
                    0.75
                )
            ),
            "max": safe_float(
                non_null.max()
            ),
            "mean": safe_float(
                non_null.mean()
            ),
        })

    return pd.DataFrame(rows)


def build_anomaly_report(
    df: pd.DataFrame,
) -> dict:
    report = {}

    if "founded" in df.columns:
        founded = pd.to_numeric(
            df["founded"],
            errors="coerce",
        )

        report[
            "founded_lt_1850"
        ] = int(
            (
                founded
                < MIN_FOUNDED_YEAR
            ).sum()
        )

        report[
            "founded_gt_2025"
        ] = int(
            (
                founded
                > MAX_FOUNDED_YEAR
            ).sum()
        )

        report[
            "founded_missing"
        ] = int(
            founded.isna().sum()
        )

    for column, key in [
        (
            "presence_world",
            "presence_world_negative",
        ),
        (
            "plans",
            "plans_negative",
        ),
        (
            "total_rented_area",
            "total_rented_area_negative",
        ),
    ]:
        if column in df.columns:
            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )
            report[key] = int(
                (numeric < 0).sum()
            )

    if (
        "total_rented_area"
        in df.columns
    ):
        area = pd.to_numeric(
            df[
                "total_rented_area"
            ],
            errors="coerce",
        )

        report[
            "total_rented_area_known_count"
        ] = int(
            area.notna().sum()
        )

        report[
            "total_rented_area_known_rate"
        ] = (
            float(
                area.notna().mean()
            )
            if len(df)
            else 0.0
        )

    if "description" in df.columns:
        description = (
            df["description"]
            .fillna("")
            .astype(str)
        )

        report[
            "description_empty"
        ] = int(
            (
                description
                .str.strip()
                == ""
            ).sum()
        )

    if "price_category" in df.columns:
        parsed = (
            df["price_category"]
            .apply(split_price_target)
        )

        known_lists = parsed.apply(
            lambda item: item[0]
        )

        report[
            "price_category_unknown_only"
        ] = int(
            (
                ~known_lists.map(bool)
            ).sum()
        )

    return report


def build_duplicate_report(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    reports = {}

    if "name" in df.columns:
        exact = (
            df.groupby(
                "name"
            )
            .size()
            .reset_index(
                name="rows"
            )
            .sort_values(
                by="rows",
                ascending=False,
            )
        )

        reports[
            "exact_name_duplicates"
        ] = exact[
            exact["rows"] > 1
        ].copy()

    if "name_clean" in df.columns:
        clean = (
            df.groupby(
                "name_clean"
            )
            .size()
            .reset_index(
                name="rows"
            )
            .sort_values(
                by="rows",
                ascending=False,
            )
        )

        reports[
            "clean_name_duplicates"
        ] = clean[
            clean["rows"] > 1
        ].copy()

        if "name" in df.columns:
            ambiguous = (
                df.groupby(
                    "name_clean"
                )["name"]
                .nunique()
                .reset_index(
                    name=(
                        "n_original_name_variants"
                    )
                )
                .sort_values(
                    by=(
                        "n_original_name_variants"
                    ),
                    ascending=False,
                )
            )

            reports[
                "ambiguous_clean_names"
            ] = ambiguous[
                ambiguous[
                    "n_original_name_variants"
                ] > 1
            ].copy()

    return reports


def build_target_reports(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    reports = {}

    if "domain" in df.columns:
        domain_counts = (
            df["domain"]
            .fillna("MISSING")
            .astype(str)
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "domain"
            )
            .reset_index(
                name="count"
            )
        )

        domain_counts["rate"] = (
            domain_counts["count"]
            / len(df)
            if len(df)
            else 0.0
        )

        reports[
            "domain_distribution"
        ] = domain_counts

    if "price_category" in df.columns:
        raw_price_counts = (
            df[
                "price_category"
            ]
            .fillna("MISSING")
            .astype(str)
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "price_category"
            )
            .reset_index(
                name="count"
            )
        )

        raw_price_counts[
            "rate"
        ] = (
            raw_price_counts["count"]
            / len(df)
            if len(df)
            else 0.0
        )

        reports[
            "price_category_distribution"
        ] = raw_price_counts

        label_rows = []

        for value in df[
            "price_category"
        ]:
            known, unexpected = (
                split_price_target(
                    value
                )
            )

            label_rows.extend(
                known
            )
            label_rows.extend(
                unexpected
            )

        if label_rows:
            label_counts = (
                pd.Series(
                    label_rows
                )
                .value_counts()
                .rename_axis(
                    "label"
                )
                .reset_index(
                    name="count"
                )
            )

            label_counts[
                "rate"
            ] = (
                label_counts[
                    "count"
                ]
                / len(df)
                if len(df)
                else 0.0
            )
        else:
            label_counts = (
                pd.DataFrame(
                    columns=[
                        "label",
                        "count",
                        "rate",
                    ]
                )
            )

        reports[
            "price_category_label_distribution"
        ] = label_counts

    if "founded" in df.columns:
        founded = pd.to_numeric(
            df["founded"],
            errors="coerce",
        )

        decades = (
            np.floor(
                founded / 10
            )
            * 10
        ).astype("Int64")

        decade_counts = (
            decades
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis(
                "decade"
            )
            .reset_index(
                name="count"
            )
        )

        decade_counts[
            "rate"
        ] = (
            decade_counts["count"]
            / max(
                1,
                founded.notna().sum(),
            )
        )

        reports[
            "founded_decade_distribution"
        ] = decade_counts

    return reports


def build_conflict_report(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    reports = {}

    key_columns = [
        column
        for column in [
            "name",
            "country_origin",
            "domain",
            "price_category",
        ]
        if column in df.columns
    ]

    if (
        len(key_columns) == 4
        and "founded" in df.columns
    ):
        conflicts = (
            df.groupby(
                key_columns
            )["founded"]
            .nunique(
                dropna=True
            )
            .reset_index(
                name=(
                    "n_founded_variants"
                )
            )
        )

        conflicts = conflicts[
            conflicts[
                "n_founded_variants"
            ] > 1
        ].copy()

        reports[
            "founded_conflicts"
        ] = conflicts.sort_values(
            by="n_founded_variants",
            ascending=False,
        )

    if (
        "name_clean" in df.columns
        and "domain" in df.columns
    ):
        domain_conflicts = (
            df.groupby(
                "name_clean"
            )["domain"]
            .nunique(
                dropna=True
            )
            .reset_index(
                name=(
                    "n_domain_variants"
                )
            )
        )

        reports[
            "name_clean_domain_conflicts"
        ] = domain_conflicts[
            domain_conflicts[
                "n_domain_variants"
            ] > 1
        ].sort_values(
            by="n_domain_variants",
            ascending=False,
        )

    if (
        "name_clean" in df.columns
        and "country_origin"
        in df.columns
    ):
        country_conflicts = (
            df.groupby(
                "name_clean"
            )["country_origin"]
            .nunique(
                dropna=True
            )
            .reset_index(
                name=(
                    "n_country_variants"
                )
            )
        )

        reports[
            "name_clean_country_conflicts"
        ] = country_conflicts[
            country_conflicts[
                "n_country_variants"
            ] > 1
        ].sort_values(
            by="n_country_variants",
            ascending=False,
        )

    return reports


def build_training_readiness_report(
    df: pd.DataFrame,
) -> dict:
    """
    Mirrors the target-eligibility rules used by the training scripts.
    QA and training therefore report the same usable populations.
    """
    report = {
        "rows_total": int(
            len(df)
        )
    }

    if "domain" in df.columns:
        non_null = df[
            df["domain"].notna()
        ].copy()

        non_null[
            "domain"
        ] = (
            non_null[
                "domain"
            ]
            .astype(str)
            .str.strip()
        )

        known = non_null[
            (non_null["domain"] != "")
            & (
                non_null["domain"]
                != UNKNOWN_DOMAIN
            )
        ].copy()

        value_counts = (
            known["domain"]
            .value_counts()
        )

        keep_classes = (
            value_counts[
                value_counts >= 5
            ]
        )

        after_rare_cut = known[
            known[
                "domain"
            ].isin(
                keep_classes.index
            )
        ]

        report["domain_task"] = {
            "rows_non_null_target": int(
                len(non_null)
            ),
            "rows_after_drop_unknown": int(
                len(known)
            ),
            "n_classes_all": int(
                value_counts.shape[0]
            ),
            "n_classes_ge_5": int(
                keep_classes.shape[0]
            ),
            "rows_after_drop_rare_classes": int(
                len(after_rare_cut)
            ),
        }

    if "founded" in df.columns:
        founded = pd.to_numeric(
            df["founded"],
            errors="coerce",
        )

        valid = founded.between(
            MIN_FOUNDED_YEAR,
            MAX_FOUNDED_YEAR,
        )

        report[
            "founded_task"
        ] = {
            "rows_non_null_target": int(
                founded.notna().sum()
            ),
            "rows_in_valid_range_1850_2025": int(
                valid.sum()
            ),
        }

    if "price_category" in df.columns:
        raw_non_null = (
            df["price_category"]
            .notna()
        )

        parsed = (
            df["price_category"]
            .apply(split_price_target)
        )

        known_lists = parsed.apply(
            lambda item: item[0]
        )

        unexpected_lists = (
            parsed.apply(
                lambda item: item[1]
            )
        )

        supports = {
            label: int(
                known_lists.apply(
                    lambda labels: (
                        label in labels
                    )
                ).sum()
            )
            for label
            in KNOWN_PRICE_LABELS
        }

        unexpected = sorted({
            label
            for labels
            in unexpected_lists
            for label in labels
        })

        known_combinations = (
            known_lists[
                known_lists.map(bool)
            ]
            .apply(
                lambda labels: ";".join(
                    labels
                )
            )
        )

        report[
            "price_category_task"
        ] = {
            "rows_non_null_target": int(
                raw_non_null.sum()
            ),
            "rows_unknown_only": int(
                (
                    ~known_lists.map(bool)
                ).sum()
            ),
            "rows_after_drop_unknown_only": int(
                known_lists.map(bool).sum()
            ),
            "n_known_labels": int(
                len(KNOWN_PRICE_LABELS)
            ),
            "known_labels": (
                KNOWN_PRICE_LABELS
            ),
            "known_label_support": (
                supports
            ),
            "min_known_label_support": (
                min(
                    supports.values()
                )
                if supports
                else 0
            ),
            "n_unique_combinations": int(
                known_combinations.nunique()
            ),
            "unexpected_labels": unexpected,
        }

    return report


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "QA checks for the retail "
            "enrichment project."
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
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(
            f"CSV file not found: "
            f"{args.csv_path}"
        )

    out_dir = args.out_dir
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")
    df_raw = pd.read_csv(
        args.csv_path
    )

    print("Harmonizing schema...")
    df_harmonized = (
        harmonize_schema(
            df_raw
        )
    )

    print(
        "Running feature preparation..."
    )
    df = prepare_features(
        df_harmonized
    )

    print(
        "Checking required columns..."
    )
    missing_required = [
        column
        for column
        in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    schema_report = (
        build_schema_report(
            df_raw,
            df,
        )
    )
    missing_report = (
        build_missing_report(
            df
        )
    )
    text_quality_report = (
        build_text_quality_report(
            df
        )
    )
    numeric_summary = (
        build_numeric_summary(
            df
        )
    )
    anomaly_report = (
        build_anomaly_report(
            df
        )
    )
    duplicate_reports = (
        build_duplicate_report(
            df
        )
    )
    target_reports = (
        build_target_reports(
            df
        )
    )
    conflict_reports = (
        build_conflict_report(
            df
        )
    )
    training_readiness = (
        build_training_readiness_report(
            df
        )
    )

    summary = {
        "dataset_shape_raw": [
            int(df_raw.shape[0]),
            int(df_raw.shape[1]),
        ],
        "dataset_shape_prepared": [
            int(df.shape[0]),
            int(df.shape[1]),
        ],
        "schema_report": (
            schema_report
        ),
        "anomaly_report": (
            anomaly_report
        ),
        "training_readiness": (
            training_readiness
        ),
        "target_contract": {
            "unknown_domain_value": (
                UNKNOWN_DOMAIN
            ),
            "unknown_price_label": (
                UNKNOWN_PRICE_LABEL
            ),
            "known_price_labels": (
                KNOWN_PRICE_LABELS
            ),
        },
    }

    if missing_required:
        summary[
            "required_columns_check"
        ] = {
            "status": "FAILED",
            "missing_required_columns": (
                missing_required
            ),
        }
    else:
        summary[
            "required_columns_check"
        ] = {
            "status": "OK",
            "missing_required_columns": [],
        }

    print("Saving reports...")

    (
        out_dir
        / "qa_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_dataframe(
        missing_report,
        out_dir
        / "missing_report.csv",
    )
    save_dataframe(
        text_quality_report,
        out_dir
        / "text_quality_report.csv",
    )
    save_dataframe(
        numeric_summary,
        out_dir
        / "numeric_summary.csv",
    )

    for name, report_df in (
        duplicate_reports.items()
    ):
        save_dataframe(
            report_df,
            out_dir / f"{name}.csv",
        )

    for name, report_df in (
        target_reports.items()
    ):
        save_dataframe(
            report_df,
            out_dir / f"{name}.csv",
        )

    for name, report_df in (
        conflict_reports.items()
    ):
        save_dataframe(
            report_df,
            out_dir / f"{name}.csv",
        )

    preview_columns = [
        column
        for column in [
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
        ]
        if column in df.columns
    ]

    save_dataframe(
        df[
            preview_columns
        ].head(1000),
        out_dir
        / "prepared_preview_1000.csv",
    )

    print(
        "\n=============================="
    )
    print("QA CHECKS COMPLETE")
    print(
        "=============================="
    )
    print(
        f"Rows raw              : "
        f"{df_raw.shape[0]}"
    )
    print(
        f"Cols raw              : "
        f"{df_raw.shape[1]}"
    )
    print(
        f"Rows prepared         : "
        f"{df.shape[0]}"
    )
    print(
        f"Cols prepared         : "
        f"{df.shape[1]}"
    )
    print(
        f"Required columns OK   : "
        f"{not bool(missing_required)}"
    )

    domain_ready = (
        training_readiness.get(
            "domain_task"
        )
    )

    if domain_ready:
        print("\n[domain]")
        print(
            "Rows with target      : "
            f"{domain_ready['rows_non_null_target']}"
        )
        print(
            "Rows after unknown cut: "
            f"{domain_ready['rows_after_drop_unknown']}"
        )
        print(
            "Classes total         : "
            f"{domain_ready['n_classes_all']}"
        )
        print(
            "Classes >= 5 rows     : "
            f"{domain_ready['n_classes_ge_5']}"
        )
        print(
            "Rows after class cut  : "
            f"{domain_ready['rows_after_drop_rare_classes']}"
        )

    founded_ready = (
        training_readiness.get(
            "founded_task"
        )
    )

    if founded_ready:
        print("\n[founded]")
        print(
            "Rows with target      : "
            f"{founded_ready['rows_non_null_target']}"
        )
        print(
            "Rows in valid range   : "
            f"{founded_ready['rows_in_valid_range_1850_2025']}"
        )

    price_ready = (
        training_readiness.get(
            "price_category_task"
        )
    )

    if price_ready:
        print(
            "\n[price_category]"
        )
        print(
            "Rows with raw target  : "
            f"{price_ready['rows_non_null_target']}"
        )
        print(
            "Unknown-only rows     : "
            f"{price_ready['rows_unknown_only']}"
        )
        print(
            "Rows usable           : "
            f"{price_ready['rows_after_drop_unknown_only']}"
        )
        print(
            "Known labels          : "
            f"{price_ready['n_known_labels']}"
        )
        print(
            "Min label support     : "
            f"{price_ready['min_known_label_support']}"
        )

    if (
        "total_rented_area_known_rate"
        in anomaly_report
    ):
        print(
            "\n[total_rented_area]"
        )
        print(
            "Known count           : "
            f"{anomaly_report['total_rented_area_known_count']}"
        )
        print(
            "Known rate            : "
            f"{anomaly_report['total_rented_area_known_rate']:.4f}"
        )

    print(
        f"\nArtifacts saved to: "
        f"{out_dir}"
    )


if __name__ == "__main__":
    main()
