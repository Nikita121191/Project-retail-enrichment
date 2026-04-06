import argparse
from pathlib import Path

import pandas as pd

from _common import harmonize_schema, normalize_country_for_audit, save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/kaggle/input/datasets/nikitasadovoy/russian-retail/russian_retail.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/kaggle/working/preprocessing_audit",
    )
    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = harmonize_schema(pd.read_csv(args.csv_path))

    if "country_origin" not in df.columns:
        raise ValueError("Column 'country_origin' not found")

    temp = df[["country_origin"]].copy()
    temp["country_origin_raw"] = temp["country_origin"].fillna("MISSING").astype(str)
    temp["country_origin_norm"] = temp["country_origin"].apply(normalize_country_for_audit)

    raw_counts = (
        temp["country_origin_raw"]
        .value_counts()
        .rename_axis("country_origin_raw")
        .reset_index(name="count")
    )

    norm_counts = (
        temp["country_origin_norm"]
        .value_counts()
        .rename_axis("country_origin_norm")
        .reset_index(name="count")
    )

    raw_to_norm = (
        temp.groupby(["country_origin_raw", "country_origin_norm"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["country_origin_norm", "count"], ascending=[True, False])
    )

    collisions = (
        raw_to_norm.groupby("country_origin_norm")["country_origin_raw"]
        .nunique()
        .reset_index(name="n_raw_variants")
        .sort_values(by="n_raw_variants", ascending=False)
    )
    collisions = collisions[collisions["n_raw_variants"] > 1].copy()

    save_csv(raw_counts, out_dir / "country_raw_value_counts.csv")
    save_csv(norm_counts, out_dir / "country_normalized_value_counts.csv")
    save_csv(raw_to_norm, out_dir / "country_raw_to_normalized_mapping_candidates.csv")
    save_csv(collisions, out_dir / "country_normalized_collision_groups.csv")

    print("audit_country done")


if __name__ == "__main__":
    main()
