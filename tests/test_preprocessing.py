import pandas as pd

from preprocessing import (
    extract_presence_values,
    prepare_features,
    region_count,
)


def test_extract_presence_with_franchise() -> None:
    own, franchise = extract_presence_values(
        "82 и 20 франчайзинговых"
    )

    assert own == 82.0
    assert franchise == 20.0


def test_extract_presence_without_franchise() -> None:
    own, franchise = extract_presence_values(
        "21"
    )

    assert own == 21.0
    assert franchise == 0.0


def test_region_count() -> None:
    result = region_count(
        "Москва; Санкт-Петербург; ; Казань"
    )

    assert result == 3


def test_prepare_features_builds_serving_features() -> None:
    df = pd.DataFrame(
        {
            "name": ["Test Brand"],
            "country_origin": ["Россия"],
            "description": [
                "Сеть основана в 2012 году, 15 магазинов"
            ],
            "presence_world": ["120"],
            "presence_russia": [
                "12 и 4 франчайзинговых"
            ],
            "presence_regions": [
                "Москва;Казань;Уфа"
            ],
            "plans": ["6"],
        }
    )

    result = prepare_features(df)

    assert "name_clean" in result.columns

    assert result.loc[
        0,
        "presence_own",
    ] == 12.0

    assert result.loc[
        0,
        "presence_franchise",
    ] == 4.0

    assert result.loc[
        0,
        "n_regions",
    ] == 3

    assert result.loc[
        0,
        "desc_has_digits",
    ] == 1

    assert result.loc[
        0,
        "desc_year_mentions",
    ] == 1

    assert result.loc[
        0,
        "presence_world",
    ] == 120

    assert result.loc[
        0,
        "plans",
    ] == 6