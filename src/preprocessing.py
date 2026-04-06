import re
from typing import List, Tuple

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

REQ_FEATURES = [
    "name",
    "description",
    "price_category",
    "country_origin",
    "domain",
    "presence_world",
    "presence_russia",
    "presence_regions",
    "plans",
    "founded",
]

BASE_FEATURE_COLUMNS = [
    "name",
    "name_clean",
    "description",
    "price_category",
    "country_origin",
    "domain",
    "presence_world",
    "presence_russia",
    "presence_own",
    "presence_franchise",
    "presence_regions",
    "n_regions",
    "plans",
    "founded",
    "desc_len",
    "desc_has_digits",
    "desc_year_mentions",
    "has_total_rented_area",
]


COUNTRY_MAPPING = {
    "россия": "Россия",
    "росия": "Россия",
    "россии": "Россия",
    "республика беларусь": "Республика Беларусь",
    "беларусь": "Республика Беларусь",
    "usa": "США",
    "америка": "США",
    "сша": "США",
}

LEGAL_FORMS = ["ооо", "зао", "ип", "оао", "ooo", "zao"]
LEGAL_FORMS_PATTERN = r"\b(?:%s)\b" % "|".join(LEGAL_FORMS)


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит схему к каноническому виду.
    Исправляет contry_origin -> country_origin.
    """
    df = df.copy()

    if "country_origin" not in df.columns and "contry_origin" in df.columns:
        df = df.rename(columns={"contry_origin": "country_origin"})

    if "country_origin" in df.columns and "contry_origin" in df.columns:
        df = df.drop(columns=["contry_origin"])

    return df


def _normalize_text(s: object) -> str:
    """
    Базовая мягкая нормализация текста.
    """
    if pd.isna(s):
        return ""

    s = str(s).strip().lower()
    s = s.replace("\u200b", "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _deduplicate_tokens_preserve_order(tokens: List[str]) -> List[str]:
    seen = set()
    out = []

    for token in tokens:
        if token not in seen:
            out.append(token)
            seen.add(token)

    return out


def normalize_name(s: object) -> str:
    """
    Нормализация названия бренда:
    - lower
    - удаление спецсимволов
    - удаление ОПФ
    - удаление повторяющихся токенов
    """
    s = _normalize_text(s)
    s = re.sub(r"[^a-zа-я0-9 ]", " ", s)
    s = re.sub(LEGAL_FORMS_PATTERN, " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = [t for t in s.split() if t]
    tokens = _deduplicate_tokens_preserve_order(tokens)

    return " ".join(tokens)


def normalize_country(s: object) -> str:
    """
    Нормализация страны происхождения.
    """
    raw = _normalize_text(s)
    raw = re.sub(r"[^\w\s,-]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    if not raw:
        return "Неизвестно"

    return COUNTRY_MAPPING.get(raw, str(s).strip())


def normalize_domain(s: object) -> str:
    """
    Мягкая нормализация domain.
    Не делаем агрессивного переписывания классов.
    """
    if pd.isna(s):
        return "Неизвестно"

    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)

    return s if s else "Неизвестно"


def normalize_price_category(s: object) -> str:
    """
    Нормализация price_category:
    - lower
    - удаление мусорных пробелов
    - замена , и : на ;
    - удаление дублей лейблов с сохранением порядка
    """
    if pd.isna(s):
        return "неизвестно"

    s = str(s).strip().lower()
    s = s.replace("\u200b", "")
    s = s.replace("\xa0", " ")
    s = s.replace(",", ";")
    s = s.replace(":", ";")

    parts = [p.strip() for p in s.split(";") if p.strip()]
    parts = _deduplicate_tokens_preserve_order(parts)

    return ";".join(parts) if parts else "неизвестно"


def extract_presence_values(text: object) -> Tuple[float, float]:
    """
    Извлекает из presence_russia:
    - presence_own
    - presence_franchise

    Примеры:
    '82 и 20 франчайзинговых' -> (82, 20)
    '21' -> (21, 0)
    """
    if pd.isna(text):
        return np.nan, np.nan

    text = str(text)
    numbers = list(map(int, re.findall(r"\d+", text)))
    low = text.lower()

    if "франчайз" in low:
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
        if len(numbers) == 1:
            return 0.0, float(numbers[0])

    if len(numbers) >= 1:
        return float(numbers[0]), 0.0

    return np.nan, np.nan


def region_count(text: object) -> int:
    """
    Считает количество регионов в presence_regions.
    """
    if pd.isna(text):
        return 0

    return len([p.strip() for p in str(text).split(";") if p.strip()])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Главная функция общей подготовки:
    - harmonize schema
    - cleaning / normalization
    - feature engineering
    """
    df = harmonize_schema(df).copy()

    # -------- text normalization --------
    if "name" in df.columns:
        df["name_clean"] = df["name"].apply(normalize_name)

    if "country_origin" in df.columns:
        df["country_origin"] = df["country_origin"].apply(normalize_country)

    if "domain" in df.columns:
        df["domain"] = df["domain"].apply(normalize_domain)

    if "price_category" in df.columns:
        df["price_category"] = df["price_category"].apply(normalize_price_category)

    # -------- numeric parsing --------
    for col in ["presence_world", "plans", "founded", "total_rented_area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------- engineered features --------
    if "presence_russia" in df.columns:
        extracted = df["presence_russia"].apply(extract_presence_values)
        df["presence_own"] = extracted.apply(lambda x: x[0])
        df["presence_franchise"] = extracted.apply(lambda x: x[1])

    if "presence_regions" in df.columns:
        df["n_regions"] = df["presence_regions"].apply(region_count)

    if "description" in df.columns:
        desc = df["description"].fillna("").astype(str)
        df["desc_len"] = desc.str.len()
        df["desc_has_digits"] = desc.str.contains(r"\d", regex=True).astype(int)
        df["desc_year_mentions"] = desc.str.count(r"\b(?:18|19|20)\d{2}\b")

    if "total_rented_area" in df.columns:
        df["has_total_rented_area"] = df["total_rented_area"].notna().astype(int)

    return df


def features_for_model(df: pd.DataFrame, target: str = "") -> pd.DataFrame:
    """
    Готовит датасет для модели и при необходимости убирает target.
    """
    df = prepare_features(df)

    if target and target in df.columns:
        df = df.drop(columns=[target], errors="ignore")

    return df


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_base_feature_columns() -> List[str]:
    return BASE_FEATURE_COLUMNS.copy()
