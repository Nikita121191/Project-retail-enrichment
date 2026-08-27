from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_contract import FEATURE_CONTRACT_VERSION


SUPPORTED_MANIFEST_VERSION = "retail_model_bundle_v1"
REQUIRED_MODELS = (
    "domain",
    "founded",
    "price_category",
)

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

class ServingContractError(RuntimeError):
    """Published model bundle cannot be safely used for serving."""

def load_manifest(
    bundle_dir: str | Path,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)

    manifest_path = (
        bundle_dir
        / "manifest.json"
    )

    if not manifest_path.is_file():
        raise ServingContractError(
            "manifest.json was not found in the "
            f"published model bundle: {manifest_path}"
        )

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

    except json.JSONDecodeError as error:
        raise ServingContractError(
            f"manifest.json is not valid JSON: {error}"
        ) from error

    except OSError as error:
        raise ServingContractError(
            f"Could not read manifest.json: {error}"
        ) from error

    if not isinstance(manifest, dict):
        raise ServingContractError(
            "manifest.json must contain a JSON object."
        )

    return manifest

def validate_published_bundle(
    bundle_dir: str | Path,
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)

    if not bundle_dir.is_dir():
        raise ServingContractError(
            "Published model bundle directory "
            f"does not exist: {bundle_dir}"
        )

    manifest = load_manifest(
        bundle_dir
    )

    manifest_version = manifest.get(
        "manifest_version"
    )

    if (
        manifest_version
        != SUPPORTED_MANIFEST_VERSION
    ):
        raise ServingContractError(
            "Unsupported manifest version: "
            f"{manifest_version!r}. "
            "Expected "
            f"{SUPPORTED_MANIFEST_VERSION!r}."
        )
    source_run_id = manifest.get(
        "source_run_id"
    )

    if (
        not isinstance(source_run_id, str)
        or not source_run_id.strip()
    ):
        raise ServingContractError(
            "Manifest has no valid source_run_id."
        )

    published_at_utc = manifest.get(
        "published_at_utc"
    )

    if (
        not isinstance(published_at_utc, str)
        or not published_at_utc.strip()
    ):
        raise ServingContractError(
            "Manifest has no valid published_at_utc."
        )

    dataset_sha256 = manifest.get(
        "dataset_sha256"
    )

    if (
        not isinstance(dataset_sha256, str)
        or not dataset_sha256.strip()
    ):
        raise ServingContractError(
            "Manifest has no valid dataset_sha256."
        )

    quality_policy_version = manifest.get(
        "model_quality_policy_version"
    )

    if (
        not isinstance(
            quality_policy_version,
            str,
        )
        or not quality_policy_version.strip()
    ):
        raise ServingContractError(
            "Manifest has no valid "
            "model_quality_policy_version."
        )

    feature_contract_version = manifest.get(
        "feature_contract_version"
    )

    if (
        feature_contract_version
        != FEATURE_CONTRACT_VERSION
    ):
        raise ServingContractError(
            "Published feature contract is "
            "incompatible with this application: "
            f"{feature_contract_version!r} != "
            f"{FEATURE_CONTRACT_VERSION!r}."
        )

    validation = manifest.get(
        "validation"
    )

    if not isinstance(validation, dict):
        raise ServingContractError(
            "Manifest validation section "
            "is missing or invalid."
        )

    if (
        validation.get(
            "model_quality_gate"
        )
        != "PASSED"
    ):
        raise ServingContractError(
            "Published bundle did not pass "
            "the model quality gate."
        )

    if (
        validation.get(
            "smoke_inference"
        )
        != "PASSED"
    ):
        raise ServingContractError(
            "Published bundle did not pass "
            "smoke inference."
        )

    models = manifest.get(
        "models"
    )

    if not isinstance(models, dict):
        raise ServingContractError(
            "Manifest models section "
            "is missing or invalid."
        )

    for model_name in REQUIRED_MODELS:
        model_info = models.get(
            model_name
        )

        if not isinstance(
            model_info,
            dict,
        ):
            raise ServingContractError(
                "Manifest is missing model entry: "
                f"{model_name}"
            )

        winner = model_info.get(
            "winner"
        )

        if (
            not isinstance(winner, str)
            or not winner.strip()
        ):
            raise ServingContractError(
                f"Model {model_name!r} "
                "has no valid winner."
            )

        relative_dir = model_info.get(
            "relative_dir"
        )

        if (
            not isinstance(relative_dir, str)
            or not relative_dir.strip()
        ):
            raise ServingContractError(
                f"Model {model_name!r} "
                "has no valid relative_dir."
            )

        if relative_dir != model_name:
            raise ServingContractError(
                f"Unexpected relative_dir for "
                f"{model_name!r}: "
                f"{relative_dir!r}."
            )

        model_dir = (
            bundle_dir
            / relative_dir
        )

        if not model_dir.is_dir():
            raise ServingContractError(
                "Published model directory "
                f"is missing: {model_dir}"
            )

        for file_name in (
            REQUIRED_SERVING_FILES[
                model_name
            ]
        ):
            file_path = (
                model_dir
                / file_name
            )

            if not file_path.is_file():
                raise ServingContractError(
                    "Required serving artifact "
                    f"is missing: {file_path}"
                )

            if file_path.stat().st_size == 0:
                raise ServingContractError(
                    "Required serving artifact "
                    f"is empty: {file_path}"
                )

    return manifest