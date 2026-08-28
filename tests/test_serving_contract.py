import json
from pathlib import Path

import pytest

from model_contract import FEATURE_CONTRACT_VERSION
from serving_contract import (
    REQUIRED_SERVING_FILES,
    SUPPORTED_MANIFEST_VERSION,
    ServingContractError,
    validate_published_bundle,
)


def write_manifest(
    bundle_dir: Path,
    manifest: dict,
) -> None:
    manifest_path = (
        bundle_dir
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


def build_valid_bundle(
    tmp_path: Path,
) -> tuple[Path, dict]:
    bundle_dir = (
        tmp_path
        / "current"
    )

    bundle_dir.mkdir()

    manifest = {
        "manifest_version": (
            SUPPORTED_MANIFEST_VERSION
        ),
        "source_run_id": "pytest-run",
        "published_at_utc": (
            "2026-01-01T00:00:00+00:00"
        ),
        "feature_contract_version": (
            FEATURE_CONTRACT_VERSION
        ),
        "dataset_sha256": "pytest-dataset-sha256",
        "model_quality_policy_version": (
            "model_quality_v1"
        ),
        "validation": {
            "model_quality_gate": "PASSED",
            "smoke_inference": "PASSED",
        },
        "models": {
            "domain": {
                "winner": "RandomForest",
                "relative_dir": "domain",
            },
            "founded": {
                "winner": "Ridge",
                "relative_dir": "founded",
            },
            "price_category": {
                "winner": "LogisticRegression",
                "relative_dir": "price_category",
            },
        },
    }

    write_manifest(
        bundle_dir,
        manifest,
    )

    for (
        model_name,
        required_files,
    ) in REQUIRED_SERVING_FILES.items():
        model_dir = (
            bundle_dir
            / model_name
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for file_name in required_files:
            (
                model_dir
                / file_name
            ).write_bytes(b"test")

    return bundle_dir, manifest


def test_valid_bundle_passes(
    tmp_path: Path,
) -> None:
    bundle_dir, manifest = (
        build_valid_bundle(tmp_path)
    )

    result = validate_published_bundle(
        bundle_dir
    )

    assert (
        result["source_run_id"]
        == manifest["source_run_id"]
    )


def test_missing_bundle_is_rejected(
    tmp_path: Path,
) -> None:
    missing_dir = (
        tmp_path
        / "does-not-exist"
    )

    with pytest.raises(
        ServingContractError
    ):
        validate_published_bundle(
            missing_dir
        )


def test_failed_quality_gate_is_rejected(
    tmp_path: Path,
) -> None:
    bundle_dir, manifest = (
        build_valid_bundle(tmp_path)
    )

    manifest["validation"][
        "model_quality_gate"
    ] = "FAILED"

    write_manifest(
        bundle_dir,
        manifest,
    )

    with pytest.raises(
        ServingContractError
    ):
        validate_published_bundle(
            bundle_dir
        )


def test_failed_smoke_inference_is_rejected(
    tmp_path: Path,
) -> None:
    bundle_dir, manifest = (
        build_valid_bundle(tmp_path)
    )

    manifest["validation"][
        "smoke_inference"
    ] = "FAILED"

    write_manifest(
        bundle_dir,
        manifest,
    )

    with pytest.raises(
        ServingContractError
    ):
        validate_published_bundle(
            bundle_dir
        )


def test_wrong_feature_contract_is_rejected(
    tmp_path: Path,
) -> None:
    bundle_dir, manifest = (
        build_valid_bundle(tmp_path)
    )

    manifest[
        "feature_contract_version"
    ] = "wrong_contract"

    write_manifest(
        bundle_dir,
        manifest,
    )

    with pytest.raises(
        ServingContractError
    ):
        validate_published_bundle(
            bundle_dir
        )


def test_missing_serving_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    bundle_dir, _ = (
        build_valid_bundle(tmp_path)
    )

    artifact_path = (
        bundle_dir
        / "domain"
        / REQUIRED_SERVING_FILES[
            "domain"
        ][0]
    )

    artifact_path.unlink()

    with pytest.raises(
        ServingContractError
    ):
        validate_published_bundle(
            bundle_dir
        )