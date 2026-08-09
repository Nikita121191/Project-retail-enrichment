import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "name",
    "country_origin",
    "domain",
    "price_category",
    "founded",
    "presence_world",
    "presence_russia",
    "presence_regions",
    "description",
    "plans",
    "total_rented_area",
]


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "country_origin" not in df.columns and "contry_origin" in df.columns:
        df = df.rename(columns={"contry_origin": "country_origin"})
    if "country_origin" in df.columns and "contry_origin" in df.columns:
        df = df.drop(columns=["contry_origin"])
    return df


def normalize_basic_text(s: object) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\u200b", "").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_name_for_audit(s: object) -> str:
    s = normalize_basic_text(s)
    s = re.sub(r"[^a-zа-яё0-9 ]", " ", s)
    s = re.sub(r"\b(?:ооо|зао|ип|оао|ooo|zao)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    words = []
    seen = set()
    for w in s.split():
        if w not in seen:
            words.append(w)
            seen.add(w)
    return " ".join(words)


def normalize_country_for_audit(s: object) -> str:
    s = normalize_basic_text(s)
    s = re.sub(r"[^\w\s,-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_domain_for_audit(s: object) -> str:
    s = normalize_basic_text(s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_price_for_audit(s: object) -> str:
    if pd.isna(s):
        return "неизвестно"
    s = str(s).strip().lower()
    s = s.replace("\u200b", "").replace("\xa0", " ")
    s = s.replace(",", ";").replace(":", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    parts = list(dict.fromkeys(parts))
    return ";".join(parts) if parts else "неизвестно"


def parse_presence_russia(text: object) -> Tuple[float, float]:
    if pd.isna(text):
        return np.nan, np.nan

    text = str(text)
    nums = list(map(int, re.findall(r"\d+", text)))
    low = text.lower()

    if "франчайз" in low:
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        if len(nums) == 1:
            return 0.0, float(nums[0])

    if len(nums) >= 1:
        return float(nums[0]), 0.0

    return np.nan, np.nan


def count_regions(text: object) -> int:
    if pd.isna(text):
        return 0
    return len([x.strip() for x in str(text).split(";") if x.strip()])


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
