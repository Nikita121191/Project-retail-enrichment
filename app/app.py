from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path("C:/Users/Никита/Documents/GitHub/NikitaSadovoy")
SRC_DIR = ROOT_DIR / "src"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Russian Retail Enrichment",
    page_icon="🛍️",
    layout="wide",
)


# -----------------------------
# Imports from project
# -----------------------------
try:
    from predict_all import run_pipeline  # type: ignore
except Exception as e:
    run_pipeline = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_sample_data(sample_path: Path) -> pd.DataFrame:
    return pd.read_csv(sample_path)


def safe_read_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Не удалось прочитать CSV: {e}") from e


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def make_count_series(df: pd.DataFrame, column: str, top_n: int | None = None) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="int64")

    series = (
        df[column]
        .fillna("неизвестно")
        .astype(str)
        .str.strip()
        .replace("", "неизвестно")
        .value_counts()
    )

    if top_n is not None:
        return series.head(top_n)

    return series


def plot_bar_series(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(series.index.astype(str), series.values)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def run_prediction(df_input: pd.DataFrame) -> pd.DataFrame:
    if run_pipeline is None:
        raise RuntimeError(
            "Не удалось импортировать run_pipeline из src/predict_all.py.\n"
            f"Детали: {IMPORT_ERROR}"
        )

    result = run_pipeline(df_input.copy(), artifacts_dir=ARTIFACTS_DIR)

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("run_pipeline должен возвращать pandas DataFrame.")

    return result


# -----------------------------
# UI
# -----------------------------
st.title("🛍️ Russian Retail Enrichment")
st.markdown(
    """
Интерактивное приложение для **обогащения данных о ритейл-брендах**.

Что делает приложение:
- принимает CSV с данными о брендах
- запускает inference pipeline
- предсказывает `domain`, `founded`, `price_category`
- показывает результат и аналитику
- позволяет скачать итоговый CSV
"""
)

with st.sidebar:
    st.header("ℹ️ Информация")
    st.write(f"**Корень проекта:** `{ROOT_DIR}`")
    st.write(f"**Папка исходного кода:** `{SRC_DIR}`")
    st.write(f"**Папка артефактов:** `{ARTIFACTS_DIR}`")

    st.markdown("---")
    st.subheader("Ожидаемые входные поля")
    st.markdown(
        """
Наиболее полезны следующие колонки:
- `name`
- `country_origin`
- `description`
- `presence_world`
- `presence_russia`
- `presence_regions`
- `plans`

`domain`, `founded`, `price_category`
могут быть пустыми — модель
попытается их дополнить.
"""
    )

    show_sample = st.checkbox("Показать пример локального датасета", value=False)

tab1, tab2 = st.tabs(["📤 Загрузка и предсказание", "ℹ️ О приложении"])

with tab1:
    uploaded_file = st.file_uploader(
        "Загрузи CSV-файл",
        type=["csv"],
        help="Файл должен быть в формате CSV",
    )

    df_source = None

    if uploaded_file is not None:
        try:
            df_source = safe_read_csv(uploaded_file)
            st.success("Файл успешно загружен.")
        except Exception as e:
            st.error(str(e))

    if show_sample:
        sample_path = DATA_DIR / "russian_retail.csv"
        if sample_path.exists():
            try:
                sample_df = load_sample_data(sample_path)
                st.markdown("### Пример локального датасета")
                st.dataframe(sample_df.head(15), use_container_width=True)
            except Exception as e:
                st.warning(f"Не удалось загрузить sample data: {e}")
        else:
            st.info("Файл `data/russian_retail.csv` не найден.")

    if df_source is not None:
        st.markdown("## Предпросмотр входных данных")

        c1, c2 = st.columns([2, 1])

        with c1:
            st.dataframe(df_source.head(20), use_container_width=True)

        with c2:
            st.metric("Строк во входном CSV", df_source.shape[0])
            st.metric("Столбцов", df_source.shape[1])
            st.markdown("**Колонки:**")
            st.write(list(df_source.columns))

        st.markdown("---")

        if st.button("🚀 Запустить предсказание", type="primary"):
            with st.spinner("Идёт запуск пайплайна..."):
                try:
                    result_df = run_prediction(df_source)
                except Exception as e:
                    st.error("Ошибка при запуске пайплайна.")
                    st.exception(e)
                else:
                    st.success("Предсказание успешно выполнено.")

                    # -----------------------------
                    # Summary metrics
                    # -----------------------------
                    domain_unique = (
                        result_df["pred_domain"].nunique(dropna=True)
                        if "pred_domain" in result_df.columns
                        else 0
                    )

                    price_unique = (
                        result_df["pred_price_category"].nunique(dropna=True)
                        if "pred_price_category" in result_df.columns
                        else 0
                    )

                    if "pred_founded_rounded" in result_df.columns:
                        median_founded = int(
                            pd.to_numeric(
                                result_df["pred_founded_rounded"], errors="coerce"
                            ).dropna().median()
                        )
                    else:
                        median_founded = 0

                    st.markdown("## Краткая сводка")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Обработано строк", result_df.shape[0])
                    m2.metric("Уникальных pred_domain", domain_unique)
                    m3.metric("Медианный estimated_founded", median_founded)
                    m4.metric("Уникальных pred_price_category", price_unique)

                    # -----------------------------
                    # Prediction-only preview
                    # -----------------------------
                    st.markdown("## Результат предсказания")
                    st.subheader("Предсказанные значения")

                    pred_cols = [
                        "name",
                        "pred_domain",
                        "pred_founded_rounded",
                        "pred_price_category",
                    ]
                    pred_cols_existing = [col for col in pred_cols if col in result_df.columns]

                    pred_only_df = result_df[pred_cols_existing].copy()
                    pred_only_df = pred_only_df.rename(
                        columns={
                            "pred_founded_rounded": "estimated_founded",
                        }
                    )

                    st.dataframe(pred_only_df.head(30), use_container_width=True)

                    st.info(
                        "⚠️ estimated_founded — это приблизительная оценка, основанная на паттернах данных, "
                        "а не точное историческое значение."
                    )

                    # -----------------------------
                    # Filled missing domain cases
                    # -----------------------------
                    st.subheader("Строки, где модель дополнила пустой domain")

                    if "domain" in result_df.columns and "pred_domain" in result_df.columns:
                        empty_domain_mask = (
                            result_df["domain"].isna()
                            | result_df["domain"].astype(str).str.strip().eq("")
                            | result_df["domain"].astype(str).str.lower().eq("none")
                            | result_df["domain"].astype(str).str.lower().eq("nan")
                        )

                        filled_domain_df = result_df.loc[
                            empty_domain_mask,
                            ["name", "domain", "pred_domain", "pred_founded_rounded", "pred_price_category"]
                        ].copy()

                        if not filled_domain_df.empty:
                            filled_domain_df = filled_domain_df.rename(
                                columns={"pred_founded_rounded": "estimated_founded"}
                            )
                            st.dataframe(filled_domain_df.head(30), use_container_width=True)
                        else:
                            st.info("Во входных данных не было пустых значений domain.")

                    # -----------------------------
                    # Full result
                    # -----------------------------
                    with st.expander("Показать полный результат"):
                        st.dataframe(result_df, use_container_width=True)

                    # -----------------------------
                    # Charts
                    # -----------------------------
                    st.markdown("## Аналитика предсказаний")

                    chart_col1, chart_col2 = st.columns(2)

                    domain_counts = make_count_series(result_df, "pred_domain", top_n=10)
                    price_counts = make_count_series(result_df, "pred_price_category", top_n=10)

                    with chart_col1:
                        if not domain_counts.empty:
                            fig_domain = plot_bar_series(
                                domain_counts,
                                title="Top-10 предсказанных категорий",
                                xlabel="pred_domain",
                                ylabel="Количество",
                            )
                            st.pyplot(fig_domain)
                            plt.close(fig_domain)
                        else:
                            st.info("Нет данных для графика `pred_domain`.")

                    with chart_col2:
                        if not price_counts.empty:
                            fig_price = plot_bar_series(
                                price_counts,
                                title="Распределение pred_price_category",
                                xlabel="pred_price_category",
                                ylabel="Количество",
                            )
                            st.pyplot(fig_price)
                            plt.close(fig_price)
                        else:
                            st.info("Нет данных для графика `pred_price_category`.")

                    # -----------------------------
                    # Top tables
                    # -----------------------------
                    t1, t2 = st.columns(2)

                    with t1:
                        st.markdown("### Топ категорий")
                        if not domain_counts.empty:
                            st.dataframe(
                                domain_counts.rename_axis("pred_domain").reset_index(name="count"),
                                use_container_width=True,
                            )

                    with t2:
                        st.markdown("### Топ ценовых сегментов")
                        if not price_counts.empty:
                            st.dataframe(
                                price_counts.rename_axis("pred_price_category").reset_index(name="count"),
                                use_container_width=True,
                            )

                    # -----------------------------
                    # Download
                    # -----------------------------
                    st.markdown("## Скачать результат")
                    st.download_button(
                        label="📥 Скачать predictions.csv",
                        data=dataframe_to_csv_bytes(result_df),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

with tab2:
    st.markdown(
        """
## О приложении

Это приложение использует уже обученные модели и запускает **batch-inference**
для CSV-файла с данными о ритейл-брендах.

### Что предсказывается
- `pred_domain` — категория бизнеса
- `pred_founded` / `pred_founded_rounded` — вероятный год основания
- `pred_price_category` — ценовой сегмент

### Когда приложение работает лучше всего
Лучшие результаты ожидаются на данных о:
- ритейл-брендах
- consumer-facing компаниях
- сетевых магазинах и схожих форматах

### Ограничения
Если загрузить бизнесы вне retail-домена, качество предсказаний может снижаться.
`pred_founded` следует интерпретировать как приблизительную оценку, а не как точный исторический факт.
"""
    )
