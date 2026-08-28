import pytest

from model_contract import (
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_MODEL_FEATURES,
    INFERENCE_SAFE_FEATURES,
    validate_feature_contract,
)


def test_feature_contract_version() -> None:
    assert FEATURE_CONTRACT_VERSION == "inference_safe_v1"


def test_inference_safe_features_contain_no_forbidden_columns() -> None:
    overlap = (
        set(INFERENCE_SAFE_FEATURES)
        & FORBIDDEN_MODEL_FEATURES
    )

    assert overlap == set()


def test_valid_feature_contract_passes() -> None:
    validate_feature_contract(
        INFERENCE_SAFE_FEATURES
    )


def test_target_column_is_rejected() -> None:
    invalid_features = [
        *INFERENCE_SAFE_FEATURES,
        "domain",
    ]

    with pytest.raises(
        ValueError,
        match="Feature contract violation",
    ):
        validate_feature_contract(
            invalid_features
        )


def test_missing_required_feature_is_rejected() -> None:
    invalid_features = [
        feature
        for feature in INFERENCE_SAFE_FEATURES
        if feature != "description"
    ]

    with pytest.raises(
        ValueError,
        match="Missing inference-safe",
    ):
        validate_feature_contract(
            invalid_features
        )