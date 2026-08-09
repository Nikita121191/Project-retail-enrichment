import argparse
from pathlib import Path

import pandas as pd

from _common import (
    harmonize_schema,
    parse_presence_russia,
    count_regions,
    save_csv,
)


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

    if "presence_russia" in df.columns:
        temp = df[["presence_russia"]].copy()
        parsed = temp["presence_russia"].apply(parse_presence_russia)
        temp["presence_own_candidate"] = parsed.apply(lambda x: x[0])
        temp["presence_franchise_candidate"] = parsed.apply(lambda x: x[1])

        summary = pd.DataFrame([{
            "rows_total": int(len(temp)),
            "rows_non_null_raw": int(temp["presence_russia"].notna().sum()),
            "presence_own_non_null": int(temp["presence_own_candidate"].notna().sum()),
            "presence_franchise_non_null": int(temp["presence_franchise_candidate"].notna().sum()),
        }])

        save_csv(summary, out_dir / "presence_russia_summary.csv")
        save_csv(temp.head(200), out_dir / "presence_russia_parsed_preview.csv")

    if "presence_regions" in df.columns:
        temp = df[["presence_regions"]].copy()
        temp["n_regions_candidate"] = temp["presence_regions"].apply(count_regions)

        summary = pd.DataFrame([{
            "rows_total": int(len(temp)),
            "rows_non_null_raw": int(temp["presence_regions"].notna().sum()),
            "mean_n_regions": float(temp["n_regions_candidate"].mean()),
            "median_n_regions": float(temp["n_regions_candidate"].median()),
            "max_n_regions": int(temp["n_regions_candidate"].max()),
        }])

        save_csv(summary, out_dir / "presence_regions_summary.csv")
        save_csv(temp.head(200), out_dir / "presence_regions_preview.csv")

    if "presence_world" in df.columns:
        s = pd.to_numeric(df["presence_world"], errors="coerce")
        summary = pd.DataFrame([{
            "rows_total": int(len(s)),
            "rows_non_null": int(s.notna().sum()),
            "negative_count": int((s < 0).sum()),
            "min": float(s.min()) if s.notna().any() else None,
            "median": float(s.median()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        }])
        save_csv(summary, out_dir / "presence_world_summary.csv")

    if "plans" in df.columns:
        s = pd.to_numeric(df["plans"], errors="coerce")
        summary = pd.DataFrame([{
            "rows_total": int(len(s)),
            "rows_non_null": int(s.notna().sum()),
            "negative_count": int((s < 0).sum()),
            "min": float(s.min()) if s.notna().any() else None,
            "median": float(s.median()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        }])
        save_csv(summary, out_dir / "plans_summary.csv")

    print("audit_presence done")


if __name__ == "__main__":
    main()
