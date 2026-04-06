import argparse
from pathlib import Path

import pandas as pd

from _common import harmonize_schema, normalize_domain_for_audit, save_csv


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

    if "domain" not in df.columns:
        raise ValueError("Column 'domain' not found")

    temp = df[["domain"]].copy()
    temp["domain_raw"] = temp["domain"].fillna("MISSING").astype(str)
    temp["domain_norm"] = temp["domain"].apply(normalize_domain_for_audit)

    raw_counts = (
        temp["domain_raw"]
        .value_counts()
        .rename_axis("domain_raw")
        .reset_index(name="count")
    )

    norm_counts = (
        temp["domain_norm"]
        .value_counts()
        .rename_axis("domain_norm")
        .reset_index(name="count")
    )

    raw_to_norm = (
        temp.groupby(["domain_raw", "domain_norm"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["domain_norm", "count"], ascending=[True, False])
    )

    save_csv(raw_counts, out_dir / "domain_raw_value_counts.csv")
    save_csv(norm_counts, out_dir / "domain_normalized_value_counts.csv")
    save_csv(raw_to_norm, out_dir / "domain_raw_to_normalized_mapping_candidates.csv")

    print("audit_domain done")


if __name__ == "__main__":
    main()
