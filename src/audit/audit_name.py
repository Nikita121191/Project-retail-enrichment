import argparse
from pathlib import Path

import pandas as pd

from _common import harmonize_schema, normalize_name_for_audit, save_csv


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

    if "name" not in df.columns:
        raise ValueError("Column 'name' not found")

    temp = df[["name"]].copy()
    temp["name_clean_candidate"] = temp["name"].apply(normalize_name_for_audit)

    raw_counts = (
        temp["name"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .rename_axis("name")
        .reset_index(name="count")
    )

    clean_counts = (
        temp["name_clean_candidate"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .rename_axis("name_clean_candidate")
        .reset_index(name="count")
    )

    ambiguous = (
        temp.groupby("name_clean_candidate")["name"]
        .nunique()
        .reset_index(name="n_original_variants")
        .sort_values(by="n_original_variants", ascending=False)
    )
    ambiguous = ambiguous[ambiguous["n_original_variants"] > 1].copy()

    examples = (
        temp.groupby("name_clean_candidate")["name"]
        .agg(lambda x: " | ".join(sorted(map(str, pd.Series(x).dropna().unique()))))
        .reset_index(name="original_examples")
    )
    examples = examples.merge(
        ambiguous,
        on="name_clean_candidate",
        how="inner"
    ).sort_values(by="n_original_variants", ascending=False)

    save_csv(raw_counts, out_dir / "name_raw_value_counts.csv")
    save_csv(clean_counts, out_dir / "name_clean_candidate_counts.csv")
    save_csv(ambiguous, out_dir / "name_clean_ambiguous_groups.csv")
    save_csv(examples, out_dir / "name_clean_ambiguous_examples.csv")

    print("audit_name done")


if __name__ == "__main__":
    main()
