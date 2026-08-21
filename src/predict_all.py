from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_contract import (
    FEATURE_CONTRACT_VERSION,
    INFERENCE_SAFE_FEATURES,
    KNOWN_PRICE_LABELS,
    MAX_FOUNDED_YEAR,
    MIN_FOUNDED_YEAR,
    validate_feature_contract,
)
from preprocessing import prepare_features


FEATURE_COLUMNS = INFERENCE_SAFE_FEATURES.copy()

TEXT_COLUMNS = [
    "description",
    "name_clean",
]

CATEGORICAL_COLUMNS = [
    "country_origin",
]

NUMERIC_COLUMNS = [
    "presence_world",
    "plans",
    "presence_own",
    "presence_franchise",
    "n_regions",
    "desc_len",
    "desc_has_digits",
    "desc_year_mentions",
]


def ensure_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for column in columns:
        if column in df.columns:
            continue

        if column in TEXT_COLUMNS:
            df[column] = ""
        elif column in CATEGORICAL_COLUMNS:
            df[column] = "Неизвестно"
        else:
            df[column] = np.nan

    return df


def sanitize_for_model(
    df: pd.DataFrame,
) -> pd.DataFrame:
    validate_feature_contract(
        FEATURE_COLUMNS
    )

    df = ensure_columns(
        df,
        FEATURE_COLUMNS,
    )

    for column in TEXT_COLUMNS:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )

    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .fillna("Неизвестно")
            .astype(str)
        )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df[
        FEATURE_COLUMNS
    ].copy()


def normalize_probability_output(
    probabilities,
) -> np.ndarray:
    if isinstance(
        probabilities,
        list,
    ):
        columns = []

        for probability in probabilities:
            probability = np.asarray(
                probability
            )

            if (
                probability.ndim == 2
                and probability.shape[1] == 2
            ):
                columns.append(
                    probability[:, 1]
                )
            else:
                columns.append(
                    probability.reshape(-1)
                )

        return np.column_stack(
            columns
        )

    probabilities = np.asarray(
        probabilities
    )

    if (
        probabilities.ndim == 3
        and probabilities.shape[2] == 2
    ):
        return probabilities[:, :, 1]

    return probabilities


def get_price_probabilities(
    fitted_model,
    X: pd.DataFrame,
) -> np.ndarray:
    if not hasattr(
        fitted_model,
        "predict_proba",
    ):
        raise TypeError(
            "price_category model does not "
            "implement predict_proba."
        )

    return normalize_probability_output(
        fitted_model.predict_proba(X)
    )


def apply_thresholds(
    y_prob: np.ndarray,
    thresholds: dict,
    labels: list[str],
) -> np.ndarray:
    if y_prob.ndim != 2:
        raise ValueError(
            "Price probabilities must be "
            "a 2D matrix."
        )

    if y_prob.shape[1] != len(labels):
        raise ValueError(
            "Probability column count does "
            "not match price labels."
        )

    if set(labels) != set(
        thresholds
    ):
        raise ValueError(
            "Price labels and threshold "
            "keys do not match."
        )

    out = np.zeros_like(
        y_prob,
        dtype=int,
    )

    for index, label in enumerate(labels):
        threshold = float(
            thresholds[label]
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for '{label}' "
                f"is outside [0, 1]: "
                f"{threshold}"
            )

        out[:, index] = (
            y_prob[:, index]
            >= threshold
        ).astype(int)

    empty_rows = np.where(
        out.sum(axis=1) == 0
    )[0]

    if len(empty_rows) > 0:
        max_index = np.argmax(
            y_prob[empty_rows],
            axis=1,
        )
        out[
            empty_rows,
            max_index,
        ] = 1

    return out


def multilabel_rows_to_strings(
    y_bin: np.ndarray,
    labels: list[str],
) -> list[str]:
    results = []

    for row in y_bin:
        active = [
            label
            for label, value
            in zip(labels, row)
            if value == 1
        ]

        results.append(
            "; ".join(active)
            if active
            else "неизвестно"
        )

    return results


def validate_price_contract(
    labels: list[str],
    thresholds: dict,
) -> None:
    if labels != KNOWN_PRICE_LABELS:
        raise ValueError(
            "price_category labels do not "
            "match the current model contract. "
            f"Expected {KNOWN_PRICE_LABELS}, "
            f"got {labels}."
        )

    if set(labels) != set(
        thresholds
    ):
        raise ValueError(
            "price_category labels and "
            "threshold keys do not match."
        )


def run_pipeline(
    df_raw: pd.DataFrame,
    artifacts_dir: str | Path,
) -> pd.DataFrame:
    """
    Scores all three enrichment targets from the same inference-safe
    feature contract. Model predictions are intentionally independent:
    predicted targets are never fed into another model.
    """
    artifacts_dir = Path(
        artifacts_dir
    )

    df = prepare_features(
        df_raw
    )
    X = sanitize_for_model(
        df
    )
    result = df_raw.copy()

    # ----------------------------------------------------
    # Domain
    # ----------------------------------------------------
    domain_dir = (
        artifacts_dir
        / "domain"
    )
    domain_model_path = (
        domain_dir
        / "domain_model.joblib"
    )
    domain_encoder_path = (
        domain_dir
        / "domain_label_encoder.joblib"
    )

    if domain_model_path.is_file():
        domain_model = joblib.load(
            domain_model_path
        )

        domain_prediction = (
            domain_model.predict(X)
        )

        if (
            np.issubdtype(
                np.asarray(
                    domain_prediction
                ).dtype,
                np.integer,
            )
            and domain_encoder_path.is_file()
        ):
            encoder = joblib.load(
                domain_encoder_path
            )
            domain_prediction = (
                encoder.inverse_transform(
                    np.asarray(
                        domain_prediction,
                        dtype=int,
                    )
                )
            )

        result[
            "pred_domain"
        ] = domain_prediction
    else:
        result[
            "pred_domain"
        ] = np.nan

    # ----------------------------------------------------
    # Founded
    # ----------------------------------------------------
    founded_model_path = (
        artifacts_dir
        / "founded"
        / "founded_model.joblib"
    )

    if founded_model_path.is_file():
        founded_model = joblib.load(
            founded_model_path
        )

        founded_prediction = np.clip(
            np.asarray(
                founded_model.predict(
                    X
                ),
                dtype=float,
            ),
            MIN_FOUNDED_YEAR,
            MAX_FOUNDED_YEAR,
        )

        result[
            "pred_founded"
        ] = founded_prediction
        result[
            "pred_founded_rounded"
        ] = np.rint(
            founded_prediction
        ).astype(int)
    else:
        result[
            "pred_founded"
        ] = np.nan
        result[
            "pred_founded_rounded"
        ] = np.nan

    # ----------------------------------------------------
    # Price category
    # ----------------------------------------------------
    price_dir = (
        artifacts_dir
        / "price_category"
    )

    price_model_path = (
        price_dir
        / "price_category_model.joblib"
    )
    price_thresholds_path = (
        price_dir
        / "price_category_thresholds.json"
    )
    price_labels_path = (
        price_dir
        / "price_category_labels.json"
    )

    price_artifacts_exist = all([
        price_model_path.is_file(),
        price_thresholds_path.is_file(),
        price_labels_path.is_file(),
    ])

    if price_artifacts_exist:
        price_model = joblib.load(
            price_model_path
        )

        with (
            price_thresholds_path.open(
                "r",
                encoding="utf-8",
            )
        ) as file:
            thresholds = json.load(
                file
            )

        with (
            price_labels_path.open(
                "r",
                encoding="utf-8",
            )
        ) as file:
            labels = json.load(
                file
            )

        validate_price_contract(
            labels,
            thresholds,
        )

        probabilities = (
            get_price_probabilities(
                price_model,
                X,
            )
        )

        binary_prediction = (
            apply_thresholds(
                probabilities,
                thresholds,
                labels,
            )
        )

        result[
            "pred_price_category"
        ] = multilabel_rows_to_strings(
            binary_prediction,
            labels,
        )

        for index, label in enumerate(labels):
            safe_label = (
                label
                .replace(" / ", "_")
                .replace(" ", "_")
                .replace("-", "_")
            )

            result[
                f"pred_price_prob__{safe_label}"
            ] = probabilities[
                :,
                index,
            ]

            result[
                f"pred_price_label__{safe_label}"
            ] = binary_prediction[
                :,
                index,
            ]
    else:
        result[
            "pred_price_category"
        ] = np.nan

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch inference for the "
            "retail enrichment models."
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
        "--out_path",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(
            f"CSV file not found: "
            f"{args.csv_path}"
        )

    if not args.artifacts_dir.is_dir():
        parser.error(
            "Artifacts directory not found: "
            f"{args.artifacts_dir}"
        )

    print("Loading raw data...")
    df_raw = pd.read_csv(
        args.csv_path
    )

    print("Running inference...")
    result = run_pipeline(
        df_raw,
        artifacts_dir=(
            args.artifacts_dir
        ),
    )

    args.out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.out_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "rows_scored": int(
            len(result)
        ),
        "feature_contract_version": (
            FEATURE_CONTRACT_VERSION
        ),
        "artifacts_dir": str(
            args.artifacts_dir
        ),
        "output_path": str(
            args.out_path
        ),
        "domain_model_used": bool(
            (
                args.artifacts_dir
                / "domain"
                / "domain_model.joblib"
            ).is_file()
        ),
        "founded_model_used": bool(
            (
                args.artifacts_dir
                / "founded"
                / "founded_model.joblib"
            ).is_file()
        ),
        "price_category_model_used": bool(
            (
                args.artifacts_dir
                / "price_category"
                / "price_category_model.joblib"
            ).is_file()
        ),
    }

    with (
        args.out_path.with_suffix(
            ".json"
        )
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n=============================="
    )
    print(
        "PREDICT ALL COMPLETE"
    )
    print(
        "=============================="
    )
    print(
        f"Rows scored      : "
        f"{len(result)}"
    )
    print(
        f"Saved to         : "
        f"{args.out_path}"
    )


if __name__ == "__main__":
    main()
