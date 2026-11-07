"""Validation of raw serving input, independent of training and model loading."""
import csv
import io
import re

import numpy as np
import pandas as pd

from preprocessing import harmonize_schema


RAW_FEATURES = (
    "name", "description", "country_origin", "presence_world",
    "presence_russia", "presence_regions", "plans",
)
MISSING_PRESENCE = {"неизвестно", "нет данных", "н/д"}


def read_inference_csv(data: bytes) -> pd.DataFrame:
    """Reject ambiguous records before pandas can rename headers or infer an index."""
    try:
        text = data.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(rows, None)
        if not header or any(not name.strip() for name in header):
            raise ValueError("CSV должен содержать непустые названия столбцов.")
        if len(set(header)) != len(header):
            raise ValueError("Названия столбцов CSV должны быть уникальными.")
        for record, row in enumerate(rows, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Запись CSV {record}: ожидалось {len(header)} полей, "
                    f"получено {len(row)}. Проверьте запятые, кавычки и пустые строки."
                )
        # Preserve textual identifiers (e.g. 001) and country codes such as NA.
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (UnicodeError, csv.Error, pd.errors.ParserError) as error:
        raise ValueError(f"Не удалось прочитать CSV в UTF-8: {error}") from error
    validate_inference_input(df)
    return df


def _bad_rows(mask: pd.Series) -> str:
    # Positions refer to data records, not potentially multiline physical lines.
    return ", ".join(str(i + 1) for i, bad in enumerate(mask) if bad)[:200]


def validate_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    """Return normalized raw fields; optional features remain optional.

    A row needs at least one observed raw feature, not any particular column.
    Missing values retain the existing training/serving imputation policy.
    """
    if df.empty:
        raise ValueError("CSV не содержит строк данных.")
    if not df.columns.is_unique:
        raise ValueError("Названия столбцов CSV должны быть уникальными.")
    if "country_origin" in df and "contry_origin" in df:
        left = df["country_origin"].fillna("").astype(str).str.strip()
        right = df["contry_origin"].fillna("").astype(str).str.strip()
        if not left.equals(right):
            raise ValueError("country_origin и contry_origin содержат разные значения.")
    # Ignore targets and extra/engineered columns when constructing model features.
    raw = harmonize_schema(df)
    raw = raw.loc[:, [c for c in RAW_FEATURES if c in raw]].copy()
    if not len(raw.columns):
        raise ValueError("Нет входных признаков модели. Допустимые столбцы: " + ", ".join(RAW_FEATURES))
    for column in raw:
        blank = raw[column].isna() | raw[column].astype(str).str.strip().eq("")
        raw[column] = raw[column].mask(blank, np.nan)
    for column in ("presence_world", "plans"):
        if column not in raw:
            continue
        values = pd.to_numeric(raw[column], errors="coerce")
        bad = raw[column].notna() & (
            values.isna() | ~np.isfinite(values) | (values < 0) | (values % 1 != 0)
        )
        if bad.any():
            raise ValueError(
                f"{column}: требуется конечное неотрицательное целое число "
                f"или пропуск. Строки данных: {_bad_rows(bad)}."
            )
        raw[column] = values
    if "presence_russia" in raw:
        values = raw["presence_russia"].copy()
        bad = pd.Series(False, index=raw.index)
        for position, value in enumerate(values):
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text.lower() in MISSING_PRESENCE:
                values.iloc[position] = np.nan
                continue
            try:
                number = float(text)
            except ValueError:
                # Existing free text extraction supports counts and franchise wording.
                invalid = (not re.search(r"\d", text)
                           or bool(re.search(r"[-−]\s*\d|\d[.,]\d|\d[eE][+-]?\d", text)))
                numbers = re.findall(r"\d+", text)
                invalid |= len(numbers) > (2 if "франчайз" in text.lower() else 1)
                invalid |= any(not np.isfinite(float(n)) for n in numbers)
            else:
                invalid = not np.isfinite(number) or number < 0 or not number.is_integer()
                if not invalid:
                    values.iloc[position] = str(int(number))
            bad.iloc[position] = invalid
        if bad.any():
            raise ValueError(
                "presence_russia: укажите целое неотрицательное количество, "
                "текст вида «12 и 4 франчайзинговых» или пропуск. "
                f"Строки данных: {_bad_rows(bad)}."
            )
        raw["presence_russia"] = values
    empty = raw.isna().all(axis=1)
    if empty.any():
        raise ValueError("Нет значений входных признаков в строках данных: " + _bad_rows(empty))
    return raw


def input_warnings(df: pd.DataFrame) -> list[str]:
    raw = validate_inference_input(df)
    missing = [c for c in RAW_FEATURES if c not in raw or raw[c].isna().any()]
    if not missing:
        return []
    return [
        "Есть отсутствующие признаки или пропуски: " + ", ".join(missing)
        + ". Используется штатная подготовка модели: пустой текст, неизвестная "
        "страна, заполнение числовых пропусков нулём. Это не подтверждает "
        "реальное отсутствие магазинов или планов; качество на таком вводе не измерено."
    ]
