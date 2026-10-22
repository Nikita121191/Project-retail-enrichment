"""Retail Enrichment UI. Install as app/app.py in the existing project."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODEL_BUNDLE_DIR = Path(
    os.getenv("MODEL_BUNDLE_DIR", str(ROOT_DIR / "artifacts" / "current"))
).expanduser()
SAMPLE_INPUT_PATH = ROOT_DIR / "app" / "sample_input.csv"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULT_KEY = "retail_prediction_result"
CONTEXT_KEY = "retail_prediction_context"
MISSING_TEXT = {"", "none", "nan", "null", "<na>"}


def clear_prediction():
    """Clear only this application's prediction, not widget state."""
    st.session_state.pop(RESULT_KEY, None)


def sync_context(context):
    """A different file or published bundle invalidates the old result."""
    if st.session_state.get(CONTEXT_KEY) != context:
        clear_prediction()
        st.session_state[CONTEXT_KEY] = context


def missing_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return (series.isna() | text.isin(MISSING_TEXT)).fillna(True)


def safe_read_csv(data: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as error:
        raise ValueError(f"Не удалось прочитать CSV: {error}") from error
    if df.empty:
        raise ValueError("CSV не содержит строк данных.")
    if not df.columns.is_unique:
        raise ValueError("Названия столбцов CSV должны быть уникальными.")
    return df


@st.cache_data
def load_sample_data(sample_path: str, modified_ns: int) -> pd.DataFrame:
    # modified_ns is part of the cache key when the example changes.
    return pd.read_csv(sample_path)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def make_count_series(df, column, top_n=None):
    if column not in df.columns:
        return pd.Series(dtype="int64")
    values = df[column].astype("string").str.strip()
    counts = values.mask(missing_mask(values), "неизвестно").value_counts()
    return counts.head(top_n) if top_n is not None else counts


def founded_column(df):
    return (
        "pred_founded_rounded"
        if "pred_founded_rounded" in df.columns
        else "pred_founded"
    )


def median_year(df):
    values = pd.to_numeric(df[founded_column(df)], errors="coerce")
    values = values.replace([float("inf"), -float("inf")], float("nan")).dropna()
    if values.empty:
        return "Нет данных"
    median = float(values.median())
    return f"{median:g}" if math.isfinite(median) else "Нет данных"


def plot_bar_series(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(series.index.astype(str), series.to_numpy())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def validate_result(result, input_rows):
    """Validate only the output fields this UI consumes."""
    if not isinstance(result, pd.DataFrame):
        raise ValueError("run_pipeline должен возвращать pandas DataFrame.")
    if not result.columns.is_unique:
        raise ValueError("В результате повторяются названия столбцов.")
    if len(result) != input_rows:
        raise ValueError(
            f"Пайплайн вернул {len(result)} строк вместо {input_rows}. "
            "Для этого интерфейса требуется один результат на входную строку."
        )
    missing = {"pred_domain", "pred_price_category"} - set(result.columns)
    if missing:
        raise ValueError(f"В результате отсутствуют столбцы: {', '.join(sorted(missing))}.")
    if not {"pred_founded", "pred_founded_rounded"} & set(result.columns):
        raise ValueError("В результате нет pred_founded или pred_founded_rounded.")


def run_prediction(df_input, pipeline):
    result = pipeline(df_input.copy(), artifacts_dir=MODEL_BUNDLE_DIR)
    validate_result(result, len(df_input))
    return result.copy()


def align_original_domain(source, result):
    """Return original domain in result order, or None if alignment is unproven.

    Never assume that matching row counts or reset indexes prove row identity.
    No extra columns are injected into the unknown inference pipeline.
    """
    if "domain" not in source.columns:
        return pd.Series(pd.NA, index=result.index, dtype="string")
    if set(source.columns).issubset(result.columns):
        preserved = result[list(source.columns)].reset_index(drop=True)
        if preserved.equals(source.reset_index(drop=True)):
            return pd.Series(source["domain"].to_numpy(), index=result.index)
    # A unique, unchanged name can be used as a key even after reordering.
    if "name" in source.columns and "name" in result.columns:
        left, right = source["name"], result["name"]
        if (not missing_mask(left).any() and not missing_mask(right).any()
                and left.is_unique and right.is_unique
                and set(left) == set(right)):
            mapping = source.set_index("name")["domain"]
            return result["name"].map(mapping)
    return None


def render_result(result, source, csv_bytes):
    year_col = founded_column(result)
    st.markdown("## Краткая сводка")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Обработано строк", len(result))
    m2.metric("Уникальных pred_domain",
              result.loc[~missing_mask(result["pred_domain"]), "pred_domain"].nunique())
    m3.metric("Медианный estimated_founded", median_year(result))
    m4.metric("Уникальных pred_price_category",
              result.loc[~missing_mask(result["pred_price_category"]), "pred_price_category"].nunique())

    for col in ("pred_domain", "pred_price_category", year_col):
        if missing_mask(result[col]).all():
            st.warning(f"В столбце {col} нет заполненных предсказаний.")

    st.markdown("## Результат предсказания")
    pred_cols = [c for c in ("name", "pred_domain", year_col, "pred_price_category")
                 if c in result.columns]
    st.dataframe(result[pred_cols].rename(columns={year_col: "estimated_founded"}).head(30),
                 use_container_width=True)
    st.info("estimated_founded — приблизительная оценка по данным, а не точный исторический факт.")

    st.subheader("Предсказания для строк с пустым исходным domain")
    original = align_original_domain(source, result)
    if original is None:
        st.warning("Не удалось однозначно сопоставить исходные строки с результатом. "
                   "Этот раздел пропущен; полный результат доступен ниже.")
    else:
        empty = missing_mask(original)
        predicted = ~missing_mask(result["pred_domain"])
        filled = empty & predicted
        if not empty.any():
            st.info("В исходных данных не было пустых значений domain.")
        else:
            st.caption(f"Пустой исходный domain: {int(empty.sum())}; "
                       f"получено предсказаний: {int(filled.sum())}.")
            if filled.any():
                display = result.loc[filled, pred_cols].copy()
                display.insert(0, "original_domain", original.loc[filled].to_numpy())
                st.dataframe(display.rename(columns={year_col: "estimated_founded"}).head(30),
                             use_container_width=True)
            else:
                st.info("Для этих строк заполненных предсказаний domain нет.")

    with st.expander("Показать полный результат"):
        st.dataframe(result, use_container_width=True)

    st.markdown("## Аналитика предсказаний")
    chart_columns = st.columns(2)
    specs = [("pred_domain", "Top-10 предсказанных категорий"),
             ("pred_price_category", "Распределение ценовых сегментов")]
    counts = [make_count_series(result, col, top_n=10) for col, _ in specs]
    for container, (col, title), series in zip(chart_columns, specs, counts):
        with container:
            if series.empty:
                st.info(f"Нет данных для графика {col}.")
            else:
                fig = plot_bar_series(series, title, col, "Количество")
                try:
                    st.pyplot(fig)
                finally:
                    plt.close(fig)
    for container, (col, title), series in zip(st.columns(2), specs, counts):
        with container:
            st.markdown(f"### {title}")
            st.dataframe(series.rename_axis(col).reset_index(name="count"),
                         use_container_width=True)

    st.markdown("## Скачать результат")
    # The normal download rerun is safe: the result lives in session_state.
    st.download_button("📥 Скачать predictions.csv", data=csv_bytes,
                       file_name="predictions.csv", mime="text/csv")


def load_runtime():
    try:
        from predict_all import run_pipeline
        from serving_contract import validate_published_bundle
        if not callable(run_pipeline):
            raise TypeError("predict_all.run_pipeline не является функцией.")
    except Exception as error:
        clear_prediction()
        st.error("Не удалось загрузить компоненты предсказания.")
        st.code(str(error))
        st.stop()

    try:
        manifest = validate_published_bundle(MODEL_BUNDLE_DIR)
        # Validate metadata used below before displaying READY.
        for target in ("domain", "founded", "price_category"):
            if not manifest["models"][target]["winner"]:
                raise ValueError(f"Не указана модель для {target}.")
        for field in ("source_run_id", "published_at_utc", "feature_contract_version"):
            if field not in manifest:
                raise ValueError(f"В manifest отсутствует {field}.")
        signature = hashlib.sha256(json.dumps(
            {"directory": str(MODEL_BUNDLE_DIR.resolve()), "manifest": manifest},
            sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")).hexdigest()
    except Exception as error:
        clear_prediction()
        st.error("Опубликованный набор моделей недоступен для предсказания.")
        st.code(str(error))
        st.stop()
    return run_pipeline, manifest, signature


def main():
    st.set_page_config(page_title="Russian Retail Enrichment", page_icon="🛍️", layout="wide")
    st.title("🛍️ Russian Retail Enrichment")
    pipeline, manifest, bundle_signature = load_runtime()
    st.markdown("Загрузите CSV с ритейл-брендами, чтобы получить предсказания "
                "категории бизнеса, года основания и ценового сегмента.")

    with st.sidebar:
        st.header("ℹ️ Информация")
        st.success("Serving status: READY")
        st.caption("Компоненты загружены, опубликованный набор проверен.")
        st.subheader("Текущие модели")
        for label, target in (("Domain", "domain"), ("Founded", "founded"),
                              ("Price category", "price_category")):
            st.write(f"**{label}:** `{manifest['models'][target]['winner']}`")
        with st.expander("Техническая информация"):
            for label, field in (("Source run", "source_run_id"),
                                 ("Published", "published_at_utc"),
                                 ("Feature contract", "feature_contract_version")):
                st.write(f"**{label}:** `{manifest[field]}`")
        st.markdown("---")
        st.subheader("Ожидаемые входные поля")
        st.markdown("Наиболее полезны: `name`, `country_origin`, `description`, "
                    "`presence_world`, `presence_russia`, `presence_regions`, `plans`.\n\n"
                    "`domain`, `founded` и `price_category` не обязательны. "
                    "Результат содержит `pred_domain`, предсказанный год и `pred_price_category`.")
        show_sample = st.checkbox("Показать пример входного CSV", value=False)

    tab1, tab2 = st.tabs(["📤 Загрузка и предсказание", "ℹ️ О приложении"])
    with tab1:
        uploaded = st.file_uploader("Загрузи CSV-файл", type=["csv"],
                                    help="CSV с разделителем-запятой, в UTF-8.",
                                    key="retail_upload", on_change=clear_prediction)
        data = uploaded.getvalue() if uploaded is not None else None
        file_signature = ((uploaded.name, hashlib.sha256(data).hexdigest())
                          if data is not None else None)
        sync_context((file_signature, bundle_signature))
        source = None
        if data is not None:
            try:
                source = safe_read_csv(data)
            except ValueError as error:
                clear_prediction()
                st.error(str(error))
        if show_sample:
            if SAMPLE_INPUT_PATH.is_file():
                try:
                    sample = load_sample_data(str(SAMPLE_INPUT_PATH), SAMPLE_INPUT_PATH.stat().st_mtime_ns)
                    st.markdown("### Пример входного CSV")
                    st.dataframe(sample, use_container_width=True)
                except Exception as error:
                    st.warning(f"Не удалось загрузить пример CSV: {error}")
            else:
                st.info("Файл app/sample_input.csv не найден.")

        if source is not None:
            st.markdown("## Предпросмотр входных данных")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(source.head(20), use_container_width=True)
            with c2:
                st.metric("Строк во входном CSV", len(source))
                st.metric("Столбцов", len(source.columns))
                st.write(list(source.columns))
            if st.button("🚀 Запустить предсказание", type="primary"):
                # On a failed retry, do not continue showing the previous success.
                clear_prediction()
                with st.spinner("Идёт запуск пайплайна..."):
                    try:
                        result = run_prediction(source, pipeline)
                        csv_bytes = dataframe_to_csv_bytes(result)
                    except Exception as error:
                        st.error("Ошибка при запуске пайплайна или проверке результата.")
                        st.exception(error)
                    else:
                        st.session_state[RESULT_KEY] = {
                            "result": result, "source": source.copy(), "csv": csv_bytes,
                        }
            # Display is independent of the current value of the run button.
            saved = st.session_state.get(RESULT_KEY)
            if saved is not None:
                st.success("Предсказание успешно выполнено.")
                try:
                    render_result(saved["result"], saved["source"], saved["csv"])
                except Exception as error:
                    st.error("Не удалось отобразить часть аналитики. Результат предсказания сохранён.")
                    with st.expander("Подробности ошибки отображения"):
                        st.exception(error)
                    st.download_button("Скачать полный результат", data=saved["csv"],
                                       file_name="predictions.csv", mime="text/csv",
                                       key="retail_fallback_download")

    with tab2:
        st.markdown("""
## О приложении
Приложение использует опубликованные модели для batch-inference по CSV.

- `pred_domain` — категория бизнеса.
- `pred_founded` / `pred_founded_rounded` — оценка года основания.
- `pred_price_category` — ценовой сегмент.

Приложение предназначено для ритейл-брендов и сетевых магазинов.
На компаниях вне retail-домена качество может снижаться. Год основания —
приблизительная оценка, а графики описывают распределение предсказаний,
не их точность.

Результат сохраняется при действиях внутри текущей сессии. Новый файл или
смена опубликованного набора моделей сбрасывают его. После перезагрузки
вкладки браузера или перезапуска сервера может потребоваться повторный расчёт.
""")


if __name__ == "__main__":
    main()
