from __future__ import annotations

from typing import Iterable

import pandas as pd


FEATURE_CONTRACT_VERSION = "inference_safe_v1"

MIN_FOUNDED_YEAR = 1850
MAX_FOUNDED_YEAR = 2025

UNKNOWN_DOMAIN = "Неизвестно"
UNKNOWN_PRICE_LABEL = "неизвестно"

KNOWN_PRICE_LABELS = [
    "дисконт",
    "ниже среднего",
    "средний",
    "выше среднего",
    "люкс / премиум",
]

# Features guaranteed to be constructible from the raw fields that the
# inference application accepts. Prediction targets are deliberately absent.
INFERENCE_SAFE_FEATURES = [
    "description",
    "name_clean",
    "country_origin",
    "presence_world",
    "plans",
    "presence_own",
    "presence_franchise",
    "n_regions",
    "desc_len",
    "desc_has_digits",
    "desc_year_mentions",
]

TEXT_FEATURES = [
    "description",
    "name_clean",
]

CATEGORICAL_FEATURES = [
    "country_origin",
]

NUMERIC_FEATURES = [
    "presence_world",
    "plans",
    "presence_own",
    "presence_franchise",
    "n_regions",
    "desc_len",
    "desc_has_digits",
    "desc_year_mentions",
]

# These columns must never enter the three enrichment models as predictors.
# The first three are prediction targets; the last two are not guaranteed
# to exist at serving time and would introduce data-availability bias.
FORBIDDEN_MODEL_FEATURES = {
    "domain",
    "founded",
    "price_category",
    "total_rented_area",
    "has_total_rented_area",
}


def validate_feature_contract(feature_columns: Iterable[str]) -> None:
    feature_columns = list(feature_columns)
    forbidden = sorted(set(feature_columns) & FORBIDDEN_MODEL_FEATURES)

    if forbidden:
        raise ValueError(
            "Feature contract violation. Target-like or serving-unsafe "
            f"columns detected: {forbidden}"
        )

    missing = [
        column
        for column in INFERENCE_SAFE_FEATURES
        if column not in feature_columns
    ]

    if missing:
        raise ValueError(
            "Feature contract violation. Missing inference-safe "
            f"features: {missing}"
        )


def normalize_price_target(value: object) -> str:
    if pd.isna(value):
        return UNKNOWN_PRICE_LABEL

    text = str(value).strip().lower()
    text = text.replace("\u200b", "")
    text = text.replace("\xa0", " ")
    text = text.replace(",", ";")
    text = text.replace(":", ";")

    parts = [part.strip() for part in text.split(";") if part.strip()]
    parts = list(dict.fromkeys(parts))

    return ";".join(parts) if parts else UNKNOWN_PRICE_LABEL


def split_price_target(
    value: object,
) -> tuple[list[str], list[str]]:
    """
    Returns (known_labels, unexpected_labels).

    UNKNOWN_PRICE_LABEL is treated as missing target information, not as a
    trainable class. Mixed values such as "средний;неизвестно" keep only the
    known label.
    """
    normalized = normalize_price_target(value)
    parts = [part.strip() for part in normalized.split(";") if part.strip()]

    known = [part for part in parts if part in KNOWN_PRICE_LABELS]
    unexpected = [
        part
        for part in parts
        if part not in KNOWN_PRICE_LABELS
        and part != UNKNOWN_PRICE_LABEL
    ]

    return known, unexpected
