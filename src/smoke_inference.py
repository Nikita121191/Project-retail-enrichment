from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_contract import (
    FEATURE_CONTRACT_VERSION,
    KNOWN_PRICE_LABELS,
    MAX_FOUNDED_YEAR,
    MIN_FOUNDED_YEAR,
)
from predict_all import run_pipeline


TARGET_COLUMNS = [
    "domain",
    "founded",
    "price_category",
]


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required smoke-test artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def safe_price_label(label: str) -> str:
    return (
        label
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    details: str,
) -> None:
    checks.append({
        "name": name,
        "passed": bool(passed),
        "details": details,
    })


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    checks: list[dict[str, Any]],
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    add_check(
        checks,
        name="required_prediction_columns",
        passed=not missing,
        details=(
            "All required prediction columns are present."
            if not missing
            else f"Missing columns: {missing}"
        ),
    )


def validate_domain_predictions(
    result: pd.DataFrame,
    artifacts_dir: Path,
    checks: list[dict[str, Any]],
) -> None:
    if "pred_domain" not in result.columns:
        return

    classes = load_json(
        artifacts_dir
        / "domain"
        / "domain_classes.json"
    )

    if not isinstance(classes, list) or not classes:
        raise ValueError(
            "domain_classes.json must contain "
            "a non-empty list."
        )

    predictions = (
        result["pred_domain"]
        .astype("string")
    )

    non_null = bool(
        predictions.notna().all()
    )
    non_empty = bool(
        predictions.fillna("")
        .str.strip()
        .ne("")
        .all()
    )

    unexpected = sorted(
        set(
            predictions
            .dropna()
            .astype(str)
        )
        - set(map(str, classes))
    )

    add_check(
        checks,
        name="domain_non_null",
        passed=non_null,
        details="All domain predictions are non-null.",
    )

    add_check(
        checks,
        name="domain_non_empty",
        passed=non_empty,
        details="All domain predictions are non-empty.",
    )

    add_check(
        checks,
        name="domain_known_classes_only",
        passed=not unexpected,
        details=(
            "All domain predictions belong to "
            "domain_classes.json."
            if not unexpected
            else f"Unexpected domain predictions: {unexpected}"
        ),
    )


def validate_founded_predictions(
    result: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    if "pred_founded" not in result.columns:
        return

    founded = pd.to_numeric(
        result["pred_founded"],
        errors="coerce",
    )

    values = founded.to_numpy(
        dtype=float
    )

    finite = bool(
        np.isfinite(values).all()
    )

    in_range = bool(
        founded.between(
            MIN_FOUNDED_YEAR,
            MAX_FOUNDED_YEAR,
        ).all()
    )

    add_check(
        checks,
        name="founded_finite",
        passed=finite,
        details="All founded predictions are finite.",
    )

    add_check(
        checks,
        name="founded_valid_range",
        passed=in_range,
        details=(
            "All founded predictions are within "
            f"[{MIN_FOUNDED_YEAR}, {MAX_FOUNDED_YEAR}]."
        ),
    )

    if "pred_founded_rounded" in result.columns:
        rounded = pd.to_numeric(
            result["pred_founded_rounded"],
            errors="coerce",
        )

        rounded_values = rounded.to_numpy(
            dtype=float
        )

        rounded_valid = bool(
            np.isfinite(
                rounded_values
            ).all()
            and np.equal(
                rounded_values,
                np.rint(rounded_values),
            ).all()
            and rounded.between(
                MIN_FOUNDED_YEAR,
                MAX_FOUNDED_YEAR,
            ).all()
        )

        add_check(
            checks,
            name="founded_rounded_valid",
            passed=rounded_valid,
            details=(
                "Rounded founded predictions are "
                "finite integers inside the valid range."
            ),
        )


def validate_price_predictions(
    result: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    if "pred_price_category" not in result.columns:
        return

    unexpected_labels: set[str] = set()
    empty_rows = 0

    for value in (
        result["pred_price_category"]
        .fillna("")
        .astype(str)
    ):
        labels = [
            part.strip()
            for part in value.split(";")
            if part.strip()
        ]

        if not labels:
            empty_rows += 1
            continue

        for label in labels:
            if label not in KNOWN_PRICE_LABELS:
                unexpected_labels.add(
                    label
                )

    add_check(
        checks,
        name="price_non_empty",
        passed=empty_rows == 0,
        details=(
            "Every row has at least one predicted "
            "price label."
            if empty_rows == 0
            else f"Rows without a price label: {empty_rows}"
        ),
    )

    add_check(
        checks,
        name="price_known_labels_only",
        passed=not unexpected_labels,
        details=(
            "All predicted price labels match "
            "the current model contract."
            if not unexpected_labels
            else (
                "Unexpected price labels: "
                f"{sorted(unexpected_labels)}"
            )
        ),
    )

    probability_columns = []
    binary_columns = []

    probability_failures = []
    binary_failures = []

    for label in KNOWN_PRICE_LABELS:
        suffix = safe_price_label(
            label
        )

        probability_column = (
            f"pred_price_prob__{suffix}"
        )
        binary_column = (
            f"pred_price_label__{suffix}"
        )

        probability_columns.append(
            probability_column
        )
        binary_columns.append(
            binary_column
        )

        if probability_column not in result.columns:
            probability_failures.append(
                f"missing {probability_column}"
            )
            continue

        probability = pd.to_numeric(
            result[probability_column],
            errors="coerce",
        )

        if (
            probability.isna().any()
            or not probability.between(
                0.0,
                1.0,
            ).all()
        ):
            probability_failures.append(
                f"invalid values in {probability_column}"
            )

        if binary_column not in result.columns:
            binary_failures.append(
                f"missing {binary_column}"
            )
            continue

        binary = pd.to_numeric(
            result[binary_column],
            errors="coerce",
        )

        if (
            binary.isna().any()
            or not binary.isin(
                [0, 1]
            ).all()
        ):
            binary_failures.append(
                f"invalid values in {binary_column}"
            )

    add_check(
        checks,
        name="price_probabilities_valid",
        passed=not probability_failures,
        details=(
            "All price probability columns exist "
            "and contain values in [0, 1]."
            if not probability_failures
            else "; ".join(
                probability_failures
            )
        ),
    )

    add_check(
        checks,
        name="price_binary_outputs_valid",
        passed=not binary_failures,
        details=(
            "All price binary label columns exist "
            "and contain only 0/1."
            if not binary_failures
            else "; ".join(
                binary_failures
            )
        ),
    )

    if not binary_failures:
        active_counts = (
            result[binary_columns]
            .astype(int)
            .sum(axis=1)
        )

        add_check(
            checks,
            name="price_at_least_one_label_per_row",
            passed=bool(
                active_counts.ge(1).all()
            ),
            details=(
                "Every row has at least one active "
                "binary price label."
            ),
        )

        mismatch_rows = 0

        for row_index, row in result.iterrows():
            expected = {
                label
                for label, column
                in zip(
                    KNOWN_PRICE_LABELS,
                    binary_columns,
                )
                if int(row[column]) == 1
            }

            observed = {
                part.strip()
                for part in str(
                    row["pred_price_category"]
                ).split(";")
                if part.strip()
            }

            if expected != observed:
                mismatch_rows += 1

        add_check(
            checks,
            name="price_text_binary_consistency",
            passed=mismatch_rows == 0,
            details=(
                "Text price predictions match the "
                "binary prediction columns."
                if mismatch_rows == 0
                else (
                    "Rows with text/binary mismatch: "
                    f"{mismatch_rows}"
                )
            ),
        )


def validate_target_independence(
    result_with_targets: pd.DataFrame,
    result_without_targets: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> None:
    """
    The three prediction targets are forbidden serving features.
    Therefore removing them from otherwise identical raw input must
    not change predictions.
    """
    domain_equal = bool(
        result_with_targets[
            "pred_domain"
        ].astype(str).equals(
            result_without_targets[
                "pred_domain"
            ].astype(str)
        )
    )

    founded_equal = bool(
        np.allclose(
            pd.to_numeric(
                result_with_targets[
                    "pred_founded"
                ],
                errors="coerce",
            ).to_numpy(dtype=float),
            pd.to_numeric(
                result_without_targets[
                    "pred_founded"
                ],
                errors="coerce",
            ).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=False,
        )
    )

    price_equal = bool(
        result_with_targets[
            "pred_price_category"
        ].astype(str).equals(
            result_without_targets[
                "pred_price_category"
            ].astype(str)
        )
    )

    probability_equal = True

    for label in KNOWN_PRICE_LABELS:
        column = (
            "pred_price_prob__"
            + safe_price_label(label)
        )

        if (
            column not in result_with_targets.columns
            or column not in result_without_targets.columns
        ):
            probability_equal = False
            break

        left = pd.to_numeric(
            result_with_targets[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        right = pd.to_numeric(
            result_without_targets[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.allclose(
            left,
            right,
            rtol=0.0,
            atol=1e-10,
            equal_nan=False,
        ):
            probability_equal = False
            break

    add_check(
        checks,
        name="target_independence_domain",
        passed=domain_equal,
        details=(
            "Domain predictions are unchanged when "
            "raw target columns are removed."
        ),
    )

    add_check(
        checks,
        name="target_independence_founded",
        passed=founded_equal,
        details=(
            "Founded predictions are unchanged when "
            "raw target columns are removed."
        ),
    )

    add_check(
        checks,
        name="target_independence_price",
        passed=(
            price_equal
            and probability_equal
        ),
        details=(
            "Price labels and probabilities are unchanged "
            "when raw target columns are removed."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an end-to-end smoke inference test "
            "against a trained retail model bundle."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(
            f"CSV file not found: {args.csv_path}"
        )

    if not args.artifacts_dir.is_dir():
        parser.error(
            "Artifacts directory not found: "
            f"{args.artifacts_dir}"
        )

    if args.sample_size <= 0:
        parser.error(
            "--sample_size must be a positive integer."
        )

    print("Loading smoke-test dataset...")
    df_raw = pd.read_csv(
        args.csv_path
    )

    if df_raw.empty:
        raise RuntimeError(
            "Smoke-test dataset is empty."
        )

    sample_size = min(
        args.sample_size,
        len(df_raw),
    )

    sample_with_targets = (
        df_raw.sample(
            n=sample_size,
            random_state=args.random_state,
        )
        .reset_index(drop=True)
    )

    # This is the important serving scenario:
    # the model must be able to enrich rows even when all three
    # prediction targets are absent from the input schema.
    sample_without_targets = (
        sample_with_targets
        .drop(
            columns=TARGET_COLUMNS,
            errors="ignore",
        )
        .copy()
    )

    print(
        f"Running inference on {sample_size} "
        "representative rows..."
    )

    result_with_targets = run_pipeline(
        sample_with_targets,
        artifacts_dir=args.artifacts_dir,
    )

    result_without_targets = run_pipeline(
        sample_without_targets,
        artifacts_dir=args.artifacts_dir,
    )

    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        name="row_count_preserved",
        passed=(
            len(result_without_targets)
            == sample_size
        ),
        details=(
            f"Input rows={sample_size}, "
            f"output rows={len(result_without_targets)}."
        ),
    )

    required_prediction_columns = [
        "pred_domain",
        "pred_founded",
        "pred_founded_rounded",
        "pred_price_category",
    ]

    require_columns(
        result_without_targets,
        required_prediction_columns,
        checks,
    )

    if all(
        column
        in result_without_targets.columns
        for column
        in required_prediction_columns
    ):
        validate_domain_predictions(
            result_without_targets,
            args.artifacts_dir,
            checks,
        )

        validate_founded_predictions(
            result_without_targets,
            checks,
        )

        validate_price_predictions(
            result_without_targets,
            checks,
        )

        validate_target_independence(
            result_with_targets,
            result_without_targets,
            checks,
        )

    failures = [
        check
        for check in checks
        if not check["passed"]
    ]

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    smoke_input_path = (
        args.out_dir
        / "smoke_input_without_targets.csv"
    )
    smoke_output_path = (
        args.out_dir
        / "smoke_predictions.csv"
    )
    report_path = (
        args.out_dir
        / "smoke_inference_report.json"
    )

    sample_without_targets.to_csv(
        smoke_input_path,
        index=False,
        encoding="utf-8-sig",
    )

    result_without_targets.to_csv(
        smoke_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "status": (
            "PASSED"
            if not failures
            else "FAILED"
        ),
        "feature_contract_version": (
            FEATURE_CONTRACT_VERSION
        ),
        "csv_path": str(
            args.csv_path
        ),
        "artifacts_dir": str(
            args.artifacts_dir
        ),
        "sample_size": int(
            sample_size
        ),
        "random_state": int(
            args.random_state
        ),
        "serving_scenario": (
            "domain, founded and price_category "
            "removed from raw inference input"
        ),
        "checks_total": len(
            checks
        ),
        "checks_passed": (
            len(checks)
            - len(failures)
        ),
        "checks_failed": len(
            failures
        ),
        "checks": checks,
        "smoke_input_path": str(
            smoke_input_path
        ),
        "smoke_output_path": str(
            smoke_output_path
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n==============================")
    print("SMOKE INFERENCE")
    print("==============================")

    for check in checks:
        marker = (
            "PASS"
            if check["passed"]
            else "FAIL"
        )
        print(
            f"[{marker}] "
            f"{check['name']}: "
            f"{check['details']}"
        )

    print("------------------------------")
    print(
        f"Result         : "
        f"{report['status']} "
        f"({report['checks_passed']}/"
        f"{report['checks_total']} "
        "checks passed)"
    )
    print(
        f"Report saved to: "
        f"{report_path}"
    )

    if failures:
        failed_names = [
            check["name"]
            for check in failures
        ]

        raise RuntimeError(
            "Smoke inference failed: "
            + ", ".join(
                failed_names
            )
        )


if __name__ == "__main__":
    main()
