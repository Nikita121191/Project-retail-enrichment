import argparse
import json
from pathlib import Path

import pandas as pd

from _common import CANONICAL_COLUMNS, harmonize_schema, save_csv


def audit_schema(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_canonical_columns": [c for c in CANONICAL_COLUMNS if c not in df.columns],
    }


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "missing_count": missing,
            "missing_rate": float(missing / n) if n else 0.0,
            "dtype": str(df[col].dtype),
        })

    return pd.DataFrame(rows).sort_values(
        by=["missing_rate", "missing_count"],
        ascending=[False, False]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv_path)
    df = harmonize_schema(df_raw)

    schema_info = audit_schema(df)
    missing_report = audit_missing(df)

    (out_dir / "schema_summary.json").write_text(
        json.dumps(schema_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_csv(missing_report, out_dir / "missing_report.csv")

    print("audit_schema_and_missing done")


if __name__ == "__main__":
    main()
