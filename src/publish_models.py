from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_DIRS = (
    "domain",
    "founded",
    "price_category",
)

REPORT_FILES = {
    "domain": "domain_report.json",
    "founded": "founded_report.json",
    "price_category": "price_category_report.json",
}

REQUIRED_SERVING_FILES = {
    "domain": (
        "domain_model.joblib",
        "domain_classes.json",
    ),
    "founded": (
        "founded_model.joblib",
    ),
    "price_category": (
        "price_category_model.joblib",
        "price_category_thresholds.json",
        "price_category_labels.json",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}"
        )

    return data


def require_passed(
    report: dict[str, Any],
    *,
    report_name: str,
) -> None:
    status = str(
        report.get("status", "")
    ).upper()

    if status != "PASSED":
        raise RuntimeError(
            f"{report_name} must have status=PASSED "
            f"before publication; got {status!r}."
        )


def get_first(
    report: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in report:
            return report[key]

    return None


def require_same_value(
    values: dict[str, Any],
    *,
    field_name: str,
) -> Any:
    missing = [
        name
        for name, value in values.items()
        if value in (None, "")
    ]

    if missing:
        raise KeyError(
            f"Missing {field_name!r} in reports: "
            f"{missing}"
        )

    unique_values = {
        str(value)
        for value in values.values()
    }

    if len(unique_values) != 1:
        raise ValueError(
            f"Inconsistent {field_name!r} across reports: "
            f"{values}"
        )

    return next(iter(values.values()))


def validate_source_bundle(
    training_dir: Path,
) -> dict[str, dict[str, Any]]:
    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {training_dir}"
        )

    reports: dict[str, dict[str, Any]] = {}

    for model_name in MODEL_DIRS:
        model_dir = (
            training_dir
            / model_name
        )

        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Required model directory not found: "
                f"{model_dir}"
            )

        for file_name in REQUIRED_SERVING_FILES[
            model_name
        ]:
            file_path = (
                model_dir
                / file_name
            )

            if (
                not file_path.is_file()
                or file_path.stat().st_size == 0
            ):
                raise FileNotFoundError(
                    "Required serving artifact missing "
                    f"or empty: {file_path}"
                )

        report_path = (
            model_dir
            / REPORT_FILES[model_name]
        )
        reports[model_name] = load_json(
            report_path
        )

    return reports


def extract_winner_name(
    report: dict[str, Any],
) -> Any:
    return get_first(
        report,
        (
            "winner_name",
            "winner_model",
            "winner",
            "selected_model",
            "best_model",
            "model_name",
        ),
    )


def quality_metrics_by_model(
    quality_report: dict[str, Any],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {
        "domain": {},
        "founded": {},
        "price_category": {},
    }

    checks = quality_report.get(
        "checks",
        [],
    )

    if not isinstance(checks, list):
        return result

    for check in checks:
        if not isinstance(check, dict):
            continue

        model = check.get("model")
        metric = check.get("metric")
        value = check.get("value")

        if (
            model in result
            and isinstance(metric, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            result[model][metric] = float(
                value
            )

    return result


def build_manifest(
    *,
    source_run_id: str,
    reports: dict[str, dict[str, Any]],
    quality_report: dict[str, Any],
    smoke_report: dict[str, Any],
) -> dict[str, Any]:
    feature_contract_version = require_same_value(
        {
            name: report.get(
                "feature_contract_version"
            )
            for name, report
            in reports.items()
        },
        field_name="feature_contract_version",
    )

    dataset_sha256 = require_same_value(
        {
            name: report.get(
                "dataset_sha256"
            )
            for name, report
            in reports.items()
        },
        field_name="dataset_sha256",
    )

    smoke_contract = smoke_report.get(
        "feature_contract_version"
    )

    if (
        smoke_contract
        and str(smoke_contract)
        != str(feature_contract_version)
    ):
        raise ValueError(
            "Smoke inference feature contract does not "
            "match the training bundle: "
            f"{smoke_contract!r} != "
            f"{feature_contract_version!r}"
        )

    quality_metrics = quality_metrics_by_model(
        quality_report
    )

    return {
        "manifest_version": "retail_model_bundle_v1",
        "source_run_id": source_run_id,
        "published_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "feature_contract_version": (
            feature_contract_version
        ),
        "dataset_sha256": dataset_sha256,
        "model_quality_policy_version": (
            quality_report.get(
                "policy_version"
            )
        ),
        "validation": {
            "model_quality_gate": (
                quality_report.get(
                    "status"
                )
            ),
            "model_quality_checks_passed": (
                quality_report.get(
                    "checks_passed"
                )
            ),
            "model_quality_checks_total": (
                quality_report.get(
                    "checks_total"
                )
            ),
            "smoke_inference": (
                smoke_report.get(
                    "status"
                )
            ),
            "smoke_checks_passed": (
                smoke_report.get(
                    "checks_passed"
                )
            ),
            "smoke_checks_total": (
                smoke_report.get(
                    "checks_total"
                )
            ),
            "smoke_sample_size": (
                smoke_report.get(
                    "sample_size"
                )
            ),
            "smoke_random_state": (
                smoke_report.get(
                    "random_state"
                )
            ),
        },
        "models": {
            model_name: {
                "winner": extract_winner_name(
                    reports[model_name]
                ),
                "relative_dir": model_name,
                "quality_metrics": (
                    quality_metrics.get(
                        model_name,
                        {},
                    )
                ),
            }
            for model_name in MODEL_DIRS
        },
        "publication_evidence": {
            "model_quality_gate_report": (
                "publication_evidence/"
                "model_quality_gate_report.json"
            ),
            "smoke_inference_report": (
                "publication_evidence/"
                "smoke_inference_report.json"
            ),
        },
    }


def create_staging_bundle(
    *,
    training_dir: Path,
    staging_dir: Path,
    quality_report_path: Path,
    smoke_report_path: Path,
    manifest: dict[str, Any],
) -> None:
    shutil.copytree(
        training_dir,
        staging_dir,
    )

    evidence_dir = (
        staging_dir
        / "publication_evidence"
    )
    evidence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        quality_report_path,
        evidence_dir
        / "model_quality_gate_report.json",
    )

    shutil.copy2(
        smoke_report_path,
        evidence_dir
        / "smoke_inference_report.json",
    )

    manifest_path = (
        staging_dir
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def validate_staging_bundle(
    staging_dir: Path,
) -> None:
    manifest_path = (
        staging_dir
        / "manifest.json"
    )

    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Staging bundle has no valid manifest.json."
        )

    for model_name in MODEL_DIRS:
        model_dir = (
            staging_dir
            / model_name
        )

        if not model_dir.is_dir():
            raise RuntimeError(
                "Staging bundle is missing model "
                f"directory: {model_dir}"
            )

        for file_name in REQUIRED_SERVING_FILES[
            model_name
        ]:
            file_path = (
                model_dir
                / file_name
            )

            if (
                not file_path.is_file()
                or file_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Staging bundle is missing serving "
                    f"artifact: {file_path}"
                )


def promote_staging_bundle(
    *,
    staging_dir: Path,
    publish_dir: Path,
) -> None:
    publish_parent = (
        publish_dir.parent
    )
    publish_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_dir = (
        publish_parent
        / (
            f".{publish_dir.name}.backup-"
            f"{uuid.uuid4().hex}"
        )
    )

    had_current = (
        publish_dir.exists()
    )

    try:
        if had_current:
            os.replace(
                publish_dir,
                backup_dir,
            )

        os.replace(
            staging_dir,
            publish_dir,
        )

    except Exception:
        # Best-effort rollback: if the old current bundle
        # was moved aside but promotion failed, restore it.
        if (
            had_current
            and backup_dir.exists()
            and not publish_dir.exists()
        ):
            os.replace(
                backup_dir,
                publish_dir,
            )

        raise

    else:
        if backup_dir.exists():
            shutil.rmtree(
                backup_dir
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publish an already validated retail model "
            "bundle to the stable current/ location."
        )
    )
    parser.add_argument(
        "--training_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--quality_report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--smoke_report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--publish_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source_run_id",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    quality_report = load_json(
        args.quality_report
    )
    smoke_report = load_json(
        args.smoke_report
    )

    require_passed(
        quality_report,
        report_name="model_quality_gate_report",
    )
    require_passed(
        smoke_report,
        report_name="smoke_inference_report",
    )

    reports = validate_source_bundle(
        args.training_dir
    )

    manifest = build_manifest(
        source_run_id=args.source_run_id,
        reports=reports,
        quality_report=quality_report,
        smoke_report=smoke_report,
    )

    publish_parent = (
        args.publish_dir.parent
    )
    publish_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = (
        publish_parent
        / (
            f".{args.publish_dir.name}.staging-"
            f"{uuid.uuid4().hex}"
        )
    )

    try:
        print(
            "Creating staging model bundle..."
        )

        create_staging_bundle(
            training_dir=args.training_dir,
            staging_dir=staging_dir,
            quality_report_path=args.quality_report,
            smoke_report_path=args.smoke_report,
            manifest=manifest,
        )

        validate_staging_bundle(
            staging_dir
        )

        print(
            "Promoting staging bundle "
            f"to {args.publish_dir}..."
        )

        promote_staging_bundle(
            staging_dir=staging_dir,
            publish_dir=args.publish_dir,
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )

    manifest_path = (
        args.publish_dir
        / "manifest.json"
    )

    print("\n==============================")
    print("PUBLISH MODELS")
    print("==============================")
    print(f"Source run     : {args.source_run_id}")
    print(f"Published dir  : {args.publish_dir}")
    print(f"Manifest       : {manifest_path}")
    print("Result         : PASSED")


if __name__ == "__main__":
    main()
