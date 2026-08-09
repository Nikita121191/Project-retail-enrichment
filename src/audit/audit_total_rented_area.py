import argparse
from pathlib import Path

import pandas as pd

from _common import harmonize_schema, save_csv


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

    if "total_rented_area" not in df.columns:
        raise ValueError("Column 'total_rented_area' not found")

    temp = df.copy()
    temp["total_rented_area_num"] = pd.to_numeric(temp["total_rented_area"], errors="coerce")
    temp["is_known_total_rented_area"] = temp["total_rented_area_num"].notna().astype(int)

    known = temp[temp["total_rented_area_num"].notna()].copy()

    summary = pd.DataFrame([{
        "rows_total": int(len(temp)),
        "rows_known": int(len(known)),
        "known_rate": float(len(known) / len(temp)) if len(temp) else 0.0,
        "rows_missing": int(temp["total_rented_area_num"].isna().sum()),
        "rows_zero": int((temp["total_rented_area_num"] == 0).sum()),
        "rows_negative": int((temp["total_rented_area_num"] < 0).sum()),
        "min": float(known["total_rented_area_num"].min()) if len(known) else None,
        "p25": float(known["total_rented_area_num"].quantile(0.25)) if len(known) else None,
        "median": float(known["total_rented_area_num"].median()) if len(known) else None,
        "p75": float(known["total_rented_area_num"].quantile(0.75)) if len(known) else None,
        "max": float(known["total_rented_area_num"].max()) if len(known) else None,
        "mean": float(known["total_rented_area_num"].mean()) if len(known) else None,
    }])

    distribution_buckets = pd.DataFrame([
        {
            "bucket": "0_1000",
            "count": int(
                (
                    (known["total_rented_area_num"] >= 0)
                    & (known["total_rented_area_num"] <= 1000)
                ).sum()
            ),
        },
        {
            "bucket": "1000_5000",
            "count": int(
                (
                    (known["total_rented_area_num"] > 1000)
                    & (known["total_rented_area_num"] <= 5000)
                ).sum()
            ),
        },
        {
            "bucket": "5000_10000",
            "count": int(
                (
                    (known["total_rented_area_num"] > 5000)
                    & (known["total_rented_area_num"] <= 10000)
                ).sum()
            ),
        },
        {
            "bucket": "10000_50000",
            "count": int(
                (
                    (known["total_rented_area_num"] > 10000)
                    & (known["total_rented_area_num"] <= 50000)
                ).sum()
            ),
        },
        {
            "bucket": "50000_100000",
            "count": int(
                (
                    (known["total_rented_area_num"] > 50000)
                    & (known["total_rented_area_num"] <= 100000)
                ).sum()
            ),
        },
        {
            "bucket": "100000_plus",
            "count": int((known["total_rented_area_num"] > 100000).sum()),
        },
    ])

    top_known_examples = pd.DataFrame()
    if len(known):
        keep_cols = [
            c for c in [
                "name",
                "country_origin",
                "domain",
                "price_category",
                "founded",
                "presence_world",
                "presence_russia",
                "presence_regions",
                "plans",
                "total_rented_area_num",
            ]
            if c in known.columns
        ]

        top_known_examples = (
            known[keep_cols]
            .sort_values(by="total_rented_area_num", ascending=False)
            .head(50)
            .copy()
        )

    missing_vs_known_by_domain = pd.DataFrame()
    if "domain" in temp.columns:
        tmp = temp.copy()
        tmp["domain"] = tmp["domain"].fillna("MISSING").astype(str)

        agg = (
            tmp.groupby("domain")["is_known_total_rented_area"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "rows_total", "sum": "rows_known"})
        )
        agg["known_rate"] = agg["rows_known"] / agg["rows_total"]
        missing_vs_known_by_domain = agg.sort_values(by="known_rate", ascending=False)

    preview = temp[[
        c for c in [
            "name",
            "domain",
            "price_category",
            "presence_world",
            "presence_russia",
            "plans",
            "total_rented_area_num",
            "is_known_total_rented_area",
        ] if c in temp.columns
    ]].head(200)

    save_csv(summary, out_dir / "total_rented_area_summary.csv")
    save_csv(distribution_buckets, out_dir / "total_rented_area_distribution_buckets.csv")
    save_csv(top_known_examples, out_dir / "total_rented_area_top_known_examples.csv")
    save_csv(missing_vs_known_by_domain, out_dir / "total_rented_area_known_rate_by_domain.csv")
    save_csv(preview, out_dir / "total_rented_area_preview.csv")

    print("audit_total_rented_area done")


if __name__ == "__main__":
    main()
