from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from model_contract import (
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_MODEL_FEATURES,
    KNOWN_PRICE_LABELS,
)


REQUIRED_FILES = {
    "domain": [
        "leaderboard.csv",
        "domain_model.joblib",
        "domain_report.json",
        "domain_classes.json",
        "domain_classification_report.txt",
        "domain_classification_report.csv",
        "domain_confusion_matrix.csv",
    ],
    "founded": [
        "leaderboard.csv",
        "founded_model.joblib",
        "founded_report.json",
        "founded_predictions_preview.csv",
    ],
    "price_category": [
        "leaderboard.csv",
        "price_category_model.joblib",
        "price_category_thresholds.json",
        "price_category_labels.json",
        "price_category_report.json",
        "price_category_classification_report.txt",
        "price_category_classification_report.csv",
    ],
}


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_nonempty_files(
    directory: Path,
    filenames: list[str],
    failures: list[str],
) -> None:
    if not directory.is_dir():
        failures.append(
            f"Directory does not exist: "
            f"{directory}"
        )
        return

    for filename in filenames:
        path = directory / filename

        if not path.is_file():
            failures.append(
                "Required artifact is missing: "
                f"{path}"
            )
            continue

        if path.stat().st_size == 0:
            failures.append(
                f"Artifact is empty: {path}"
            )


def validate_model_load(
    path: Path,
    failures: list[str],
) -> None:
    try:
        model = joblib.load(path)
    except Exception as error:
        failures.append(
            f"Could not deserialize model "
            f"{path}: {error}"
        )
        return

    if not hasattr(model, "predict"):
        failures.append(
            f"Deserialized object has no "
            f"predict() method: {path}"
        )


def validate_feature_columns(
    report: dict,
    model_name: str,
    failures: list[str],
) -> None:
    feature_columns = report.get(
        "feature_columns"
    )

    if not isinstance(
        feature_columns,
        list,
    ) or not feature_columns:
        failures.append(
            f"{model_name}: feature_columns "
            "must be a non-empty list."
        )
        return

    forbidden = sorted(
        set(feature_columns)
        & FORBIDDEN_MODEL_FEATURES
    )

    if forbidden:
        failures.append(
            f"{model_name}: serving-unsafe "
            f"features detected: {forbidden}"
        )

    if (
        report.get(
            "feature_contract_version"
        )
        != FEATURE_CONTRACT_VERSION
    ):
        failures.append(
            f"{model_name}: unexpected "
            "feature_contract_version."
        )


def validate_artifacts(
    domain_dir: Path,
    founded_dir: Path,
    price_dir: Path,
) -> None:
    failures = []

    validate_nonempty_files(
        domain_dir,
        REQUIRED_FILES["domain"],
        failures,
    )
    validate_nonempty_files(
        founded_dir,
        REQUIRED_FILES["founded"],
        failures,
    )
    validate_nonempty_files(
        price_dir,
        REQUIRED_FILES[
            "price_category"
        ],
        failures,
    )

    if failures:
        print(
            "\n=== ARTIFACT VALIDATION ==="
        )

        for failure in failures:
            print(
                f"FAILED: {failure}"
            )

        raise RuntimeError(
            "Required training artifacts "
            "are missing or empty."
        )

    domain_model_path = (
        domain_dir
        / "domain_model.joblib"
    )
    founded_model_path = (
        founded_dir
        / "founded_model.joblib"
    )
    price_model_path = (
        price_dir
        / "price_category_model.joblib"
    )

    validate_model_load(
        domain_model_path,
        failures,
    )
    validate_model_load(
        founded_model_path,
        failures,
    )
    validate_model_load(
        price_model_path,
        failures,
    )

    domain_report = load_json(
        domain_dir
        / "domain_report.json"
    )
    founded_report = load_json(
        founded_dir
        / "founded_report.json"
    )
    price_report = load_json(
        price_dir
        / "price_category_report.json"
    )

    validate_feature_columns(
        domain_report,
        "domain",
        failures,
    )
    validate_feature_columns(
        founded_report,
        "founded",
        failures,
    )
    validate_feature_columns(
        price_report,
        "price_category",
        failures,
    )

    if (
        domain_report.get("target")
        != "domain"
    ):
        failures.append(
            "domain_report.json has "
            "invalid target."
        )

    if (
        domain_report.get(
            "selection_metric"
        )
        != "cv_best_macro_f1"
    ):
        failures.append(
            "Domain model has unexpected "
            "selection metric."
        )

    if not domain_report.get(
        "winner_name"
    ):
        failures.append(
            "Domain report has no "
            "winner_name."
        )

    for key in [
        "cv_best_macro_f1",
        "cv_std_macro_f1",
        "holdout_macro_f1",
        "holdout_balanced_accuracy",
        "dataset_sha256",
    ]:
        if key not in domain_report:
            failures.append(
                f"Domain report missing: {key}"
            )

    domain_classes = load_json(
        domain_dir
        / "domain_classes.json"
    )

    if not isinstance(
        domain_classes,
        list,
    ):
        failures.append(
            "domain_classes.json must "
            "contain a list."
        )
    elif len(domain_classes) < 2:
        failures.append(
            "Domain model must contain "
            "at least two classes."
        )

    if (
        founded_report.get("target")
        != "founded"
    ):
        failures.append(
            "founded_report.json has "
            "invalid target."
        )

    if (
        founded_report.get(
            "selection_metric"
        )
        != "cv_best_mae"
    ):
        failures.append(
            "Founded model has unexpected "
            "selection metric."
        )

    if not founded_report.get(
        "winner_name"
    ):
        failures.append(
            "Founded report has no "
            "winner_name."
        )

    for key in [
        "cv_best_mae",
        "cv_std_mae",
        "holdout_mae",
        "holdout_rmse",
        "holdout_r2",
        "dataset_sha256",
    ]:
        if key not in founded_report:
            failures.append(
                f"Founded report missing: {key}"
            )

    if (
        price_report.get("target")
        != "price_category"
    ):
        failures.append(
            "price_category_report.json "
            "has invalid target."
        )

    if (
        price_report.get(
            "selection_metric"
        )
        != "cv_best_macro_f1"
    ):
        failures.append(
            "Price model has unexpected "
            "selection metric."
        )

    if (
        price_report.get(
            "threshold_calibration"
        )
        != "development_oof_cv"
    ):
        failures.append(
            "Price model has unexpected "
            "threshold calibration protocol."
        )

    if not price_report.get(
        "winner_name"
    ):
        failures.append(
            "Price report has no "
            "winner_name."
        )

    for key in [
        "cv_best_macro_f1",
        "cv_std_macro_f1",
        "oof_selected_macro_f1",
        "holdout_serving_macro_f1",
        "holdout_serving_micro_f1",
        "dataset_sha256",
    ]:
        if key not in price_report:
            failures.append(
                f"Price report missing: {key}"
            )

    labels = load_json(
        price_dir
        / "price_category_labels.json"
    )

    thresholds = load_json(
        price_dir
        / "price_category_thresholds.json"
    )

    if labels != KNOWN_PRICE_LABELS:
        failures.append(
            "Price labels do not match the "
            "known-label contract."
        )

    if not isinstance(
        thresholds,
        dict,
    ):
        failures.append(
            "price_category_thresholds.json "
            "must contain an object."
        )
    elif set(labels) != set(
        thresholds
    ):
        failures.append(
            "Price labels and threshold "
            "keys do not match."
        )
    else:
        for label, threshold in (
            thresholds.items()
        ):
            if not isinstance(
                threshold,
                (int, float),
            ):
                failures.append(
                    f"Threshold for '{label}' "
                    "is not numeric."
                )
                continue

            if not 0.0 <= threshold <= 1.0:
                failures.append(
                    f"Threshold for '{label}' "
                    "is outside [0, 1]."
                )

    fingerprints = {
        domain_report.get(
            "dataset_sha256"
        ),
        founded_report.get(
            "dataset_sha256"
        ),
        price_report.get(
            "dataset_sha256"
        ),
    }

    fingerprints.discard(None)

    if len(fingerprints) != 1:
        failures.append(
            "Training artifacts were not "
            "produced from the same dataset "
            "fingerprint."
        )

    print(
        "\n=== ARTIFACT VALIDATION REPORT ==="
    )

    if failures:
        for failure in failures:
            print(
                f"FAILED: {failure}"
            )

        raise RuntimeError(
            "Training artifact validation "
            "failed."
        )

    print(
        "Domain artifacts: OK"
    )
    print(
        "Founded artifacts: OK"
    )
    print(
        "Price category artifacts: OK"
    )
    print(
        "Model deserialization: OK"
    )
    print(
        "Feature contract: OK"
    )
    print(
        "Dataset fingerprint: OK"
    )
    print(
        "\nALL TRAINING ARTIFACTS "
        "ARE VALID."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the training artifact "
            "contract before model promotion."
        )
    )

    parser.add_argument(
        "--domain_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--founded_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--price_dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    validate_artifacts(
        domain_dir=args.domain_dir,
        founded_dir=args.founded_dir,
        price_dir=args.price_dir,
    )


if __name__ == "__main__":
    main()
