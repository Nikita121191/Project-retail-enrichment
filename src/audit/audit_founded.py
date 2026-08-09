import argparse
from pathlib import Path

import pandas as pd

from _common import harmonize_schema, save_csv


MIN_FOUNDED_YEAR = 1850
MAX_FOUNDED_YEAR = 2025


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file not found: {args.csv_path}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = harmonize_schema(pd.read_csv(args.csv_path))

    if "founded" not in df.columns:
        raise ValueError("Column 'founded' not found")

    temp = df.copy()
    temp["founded_num"] = pd.to_numeric(temp["founded"], errors="coerce")

    summary = pd.DataFrame([{
        "rows_total": int(len(temp)),
        "rows_non_null": int(temp["founded_num"].notna().sum()),
        "rows_lt_1850": int((temp["founded_num"] < MIN_FOUNDED_YEAR).sum()),
        "rows_gt_2025": int((temp["founded_num"] > MAX_FOUNDED_YEAR).sum()),
        "min": float(temp["founded_num"].min()) if temp["founded_num"].notna().any() else None,
        "median": float(temp["founded_num"].median()) if temp["founded_num"].notna().any() else None,
        "max": float(temp["founded_num"].max()) if temp["founded_num"].notna().any() else None,
    }])

    invalid_rows = temp[
        (temp["founded_num"] < MIN_FOUNDED_YEAR) | (temp["founded_num"] > MAX_FOUNDED_YEAR)
    ].copy()
    keep_cols = [
        c
        for c in [
            "name",
            "country_origin",
            "domain",
            "price_category",
            "founded",
            "founded_num",
        ]
        if c in invalid_rows.columns
    ]
    invalid_rows = invalid_rows[keep_cols].sort_values(by="founded_num")

    conflicts = pd.DataFrame()
    if all(c in temp.columns for c in ["name", "country_origin", "domain", "price_category"]):
        conflicts = (
            temp.groupby(["name", "country_origin", "domain", "price_category"])["founded_num"]
            .nunique(dropna=True)
            .reset_index(name="n_founded_variants")
            .sort_values(by="n_founded_variants", ascending=False)
        )
        conflicts = conflicts[conflicts["n_founded_variants"] > 1].copy()

    save_csv(summary, out_dir / "founded_summary.csv")
    save_csv(invalid_rows, out_dir / "founded_out_of_range_rows.csv")
    save_csv(conflicts, out_dir / "founded_conflicts.csv")

    print("audit_founded done")


if __name__ == "__main__":
    main()
