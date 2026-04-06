import argparse
from pathlib import Path

import pandas as pd

from _common import harmonize_schema, save_csv


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

    if "description" not in df.columns:
        raise ValueError("Column 'description' not found")

    temp = df[["description"]].copy()
    temp["description_text"] = temp["description"].fillna("").astype(str).str.strip()

    temp["desc_len"] = temp["description_text"].str.len()
    temp["desc_word_count"] = temp["description_text"].str.split().apply(len)
    temp["desc_has_digits"] = temp["description_text"].str.contains(r"\d", regex=True).astype(int)
    temp["desc_year_mentions"] = temp["description_text"].str.count(r"\b(?:18|19|20)\d{2}\b")
    temp["desc_is_empty"] = (temp["description_text"] == "").astype(int)

    summary = pd.DataFrame([{
        "rows_total": int(len(temp)),
        "rows_empty": int(temp["desc_is_empty"].sum()),
        "empty_rate": float(temp["desc_is_empty"].mean()),
        "mean_len": float(temp["desc_len"].mean()),
        "median_len": float(temp["desc_len"].median()),
        "max_len": int(temp["desc_len"].max()),
        "mean_word_count": float(temp["desc_word_count"].mean()),
        "median_word_count": float(temp["desc_word_count"].median()),
        "rows_with_digits": int(temp["desc_has_digits"].sum()),
        "rows_with_year_mentions": int((temp["desc_year_mentions"] > 0).sum()),
    }])

    length_distribution = pd.DataFrame([
        {
            "bucket": "0",
            "count": int((temp["desc_len"] == 0).sum()),
        },
        {
            "bucket": "1_50",
            "count": int(((temp["desc_len"] >= 1) & (temp["desc_len"] <= 50)).sum()),
        },
        {
            "bucket": "51_150",
            "count": int(((temp["desc_len"] >= 51) & (temp["desc_len"] <= 150)).sum()),
        },
        {
            "bucket": "151_300",
            "count": int(((temp["desc_len"] >= 151) & (temp["desc_len"] <= 300)).sum()),
        },
        {
            "bucket": "301_600",
            "count": int(((temp["desc_len"] >= 301) & (temp["desc_len"] <= 600)).sum()),
        },
        {
            "bucket": "600_plus",
            "count": int((temp["desc_len"] > 600).sum()),
        },
    ])

    shortest_nonempty = (
        temp[temp["desc_len"] > 0]
        .sort_values(by="desc_len", ascending=True)
        .head(50)
        .copy()
    )

    longest_examples = (
        temp.sort_values(by="desc_len", ascending=False)
        .head(50)
        .copy()
    )

    year_mentions_distribution = (
        temp["desc_year_mentions"]
        .value_counts()
        .sort_index()
        .rename_axis("year_mentions_count")
        .reset_index(name="rows")
    )

    preview = temp[[
        "description_text",
        "desc_len",
        "desc_word_count",
        "desc_has_digits",
        "desc_year_mentions",
        "desc_is_empty",
    ]].head(200)

    save_csv(summary, out_dir / "description_summary.csv")
    save_csv(length_distribution, out_dir / "description_length_distribution.csv")
    save_csv(shortest_nonempty, out_dir / "description_shortest_examples.csv")
    save_csv(longest_examples, out_dir / "description_longest_examples.csv")
    save_csv(year_mentions_distribution, out_dir / "description_year_mentions_distribution.csv")
    save_csv(preview, out_dir / "description_preview.csv")

    print("audit_description done")


if __name__ == "__main__":
    main()
