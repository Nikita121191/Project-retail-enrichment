import argparse
from pathlib import Path

import pandas as pd

from _common import harmonize_schema, normalize_price_for_audit, save_csv


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

    if "price_category" not in df.columns:
        raise ValueError("Column 'price_category' not found")

    temp = df[["price_category"]].copy()
    temp["price_category_raw"] = temp["price_category"].fillna("MISSING").astype(str)
    temp["price_category_norm"] = temp["price_category"].apply(normalize_price_for_audit)

    raw_counts = (
        temp["price_category_raw"]
        .value_counts()
        .rename_axis("price_category_raw")
        .reset_index(name="count")
    )

    norm_counts = (
        temp["price_category_norm"]
        .value_counts()
        .rename_axis("price_category_norm")
        .reset_index(name="count")
    )

    label_rows = []
    for value in temp["price_category_norm"]:
        parts = [p.strip() for p in str(value).split(";") if p.strip()]
        for p in parts:
            label_rows.append(p)

    label_dist = (
        pd.Series(label_rows)
        .value_counts()
        .rename_axis("price_label")
        .reset_index(name="count")
        if label_rows else pd.DataFrame(columns=["price_label", "count"])
    )

    save_csv(raw_counts, out_dir / "price_category_raw_value_counts.csv")
    save_csv(norm_counts, out_dir / "price_category_normalized_value_counts.csv")
    save_csv(label_dist, out_dir / "price_category_label_distribution.csv")

    print("audit_price_category done")


if __name__ == "__main__":
    main()
