import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from input_validation import input_warnings, read_inference_csv, validate_inference_input
from predict_all import run_pipeline


@pytest.mark.parametrize("data, message", [
    (b"", "названия"),
    (b"name,description\n", "строк данных"),
    (b"name,name\nA,B\n", "уникальными"),
    (b"name,\nA,B\n", "названия"),
    (b"name,description\nA,B,C\n", "полей"),
    (b"name,description\nA\n", "полей"),
    (b"name\nA\n\nB\n", "полей"),
    (b'name,description\nA,"broken\n', "прочитать"),
    (b"name\n\xff\n", "UTF-8"),
    (b"custom\nhello\n", "Нет входных"),
    (b"name;description\nA;shop\n", "Нет входных"),
    (b"name,description\nA,shop\n,\n", "строках данных: 2"),
    (b"name,description\n   , \n", "Нет значений"),
    (b"name,country_origin,contry_origin\nA,USA,Russia\n", "разные значения"),
])
def test_reject_ambiguous_or_empty_csv(data, message):
    with pytest.raises(ValueError, match=message):
        read_inference_csv(data)


@pytest.mark.parametrize("blank", [b"   ", b"\t", b" \t "])
def test_whitespace_record_is_rejected_with_record_number(blank):
    with pytest.raises(ValueError, match=r"Запись CSV 3:"):
        read_inference_csv(b"name\nA\n" + blank + b"\nB\n")


@pytest.mark.parametrize("data", [
    b"name,presence_world\nBrand,12\x00oops\n",
    b"na\x00me\nBrand\n",
    b'name,description\nBrand,"text\x00tail"\n',
])
def test_nul_is_rejected_before_csv_parsing(data, monkeypatch):
    def unexpected_parse(*args, **kwargs):
        pytest.fail("NUL input must be rejected before pandas parsing")

    monkeypatch.setattr(pd, "read_csv", unexpected_parse)
    with pytest.raises(ValueError, match="нулевой байт"):
        read_inference_csv(data)


@pytest.mark.parametrize("column", ["presence_world", "plans"])
@pytest.mark.parametrize("value", ["abc", "inf", "-inf", "NaN", "-1", "1.5", "1e999"])
def test_bad_numeric_values_are_not_silently_imputed(column, value):
    with pytest.raises(ValueError, match=column):
        validate_inference_input(pd.DataFrame({"name": ["Brand"], column: [value]}))


@pytest.mark.parametrize("value", ["-12", "12.5", "10-20", "abc", "inf", "12 и 4"])
def test_reject_ambiguous_presence_counts(value):
    with pytest.raises(ValueError, match="presence_russia"):
        validate_inference_input(pd.DataFrame({"name": ["Brand"], "presence_russia": [value]}))


@pytest.mark.parametrize("value", [
    "10–20 франчайзинговых", "12/5 франчайзинговых",
    "10—20 франчайзинговых", "12 / 5 франчайзинговых",
    "от 10 до 20 франчайзинговых", "10:20 франчайзинговых",
])
def test_two_presence_counts_require_an_unambiguous_format(value):
    with pytest.raises(ValueError, match="presence_russia"):
        read_inference_csv(f"name,presence_russia\nBrand,{value}\n".encode())
    with pytest.raises(ValueError, match="presence_russia"):
        validate_inference_input(pd.DataFrame({"name": ["Brand"], "presence_russia": [value]}))


@pytest.mark.parametrize("value, own, franchise", [
    ("80 и 876 франчайзинговых", 80, 876),
    ("82 и 20 франчайзинговых", 82, 20),
    ("12 и 4 франчайзинговых", 12, 4),
    ("80 собственных и 876 франчайзинговых", 80, 876),
    ("21", 21, 0), ("21.0", 21, 0), ("1e3", 1000, 0),
    ("0", 0, 0), ("21 магазин", 21, 0),
    ("20 франчайзинговых", 0, 20),
])
def test_documented_presence_formats_keep_their_meaning(value, own, franchise):
    from preprocessing import prepare_features

    df = read_inference_csv(f"name,presence_russia\nBrand,{value}\n".encode())
    features = prepare_features(validate_inference_input(df))
    assert features.loc[0, "presence_own"] == own
    assert features.loc[0, "presence_franchise"] == franchise


def test_preserve_identifiers_country_codes_bom_and_multiline_fields():
    df = read_inference_csv('\ufeffname,description,country_origin\r\n001,"shop, clothes\nnew line",NA\r\n'.encode())
    assert df.loc[0, "name"] == "001"
    assert df.loc[0, "country_origin"] == "NA"
    assert df.loc[0, "description"] == "shop, clothes\nnew line"
    assert len(df) == 1


@pytest.mark.parametrize("data", [b"name\nBrand\n", b"description\nShop\n",
    b"presence_world\n0\n", b"country_origin\nNewCountry\n"])
def test_no_single_raw_column_is_required(data):
    df = read_inference_csv(data)
    assert len(validate_inference_input(df)) == 1
    assert input_warnings(df)


def test_missing_values_are_supported_and_disclosed_without_dropping_rows():
    df = read_inference_csv(b"name,description,plans,presence_russia\nA,   , ,\nB,,0,21.0\n")
    result = validate_inference_input(df)
    assert len(result) == 2
    assert result["description"].isna().all()
    assert np.isnan(result.loc[0, "plans"])
    assert result.loc[1, "plans"] == 0
    assert result.loc[1, "presence_russia"] == "21"
    assert input_warnings(df)


def test_targets_extras_and_engineered_columns_cannot_supply_model_features():
    df = pd.DataFrame({"name": ["Brand"], "domain": ["fake"], "founded": ["bad"],
                       "price_category": ["fake"], "desc_len": [999], "custom": ["extra"]})
    assert list(validate_inference_input(df)) == ["name"]
    # The direct inference API must validate before opening model files too.
    with pytest.raises(ValueError, match="Нет входных"):
        run_pipeline(df.drop(columns="name"), Path("nonexistent-models"))


def test_alias_and_column_order_are_supported():
    a = pd.DataFrame({"name": ["Brand"], "contry_origin": ["Россия"], "plans": ["2"]})
    pd.testing.assert_frame_equal(validate_inference_input(a), validate_inference_input(a[a.columns[::-1]]))
    assert validate_inference_input(a).loc[0, "country_origin"] == "Россия"


def test_inference_recomputes_features_and_preserves_extra_output_columns(tmp_path, monkeypatch):
    import predict_all

    directory = tmp_path / "domain"
    directory.mkdir()
    (directory / "domain_model.joblib").touch()
    seen = []

    class Model:
        def predict(self, features):
            seen.append(features.copy())
            return np.array(["Одежда"])

    monkeypatch.setattr(predict_all.joblib, "load", lambda path: Model())
    data = pd.DataFrame({"name": ["Brand"], "desc_len": [99999],
                         "name_clean": ["injected"], "customer_id": ["001"]})
    result = run_pipeline(data, tmp_path)
    assert seen[0].loc[0, "name_clean"] == "brand"
    assert pd.isna(seen[0].loc[0, "desc_len"])
    assert result.loc[0, "customer_id"] == "001"
    assert result.loc[0, "desc_len"] == data.loc[0, "desc_len"] == 99999


def test_ui_uses_strict_reader_and_discards_results_when_context_changes(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "app" / "app.py"
    spec = importlib.util.spec_from_file_location("retail_ui", path)
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    with pytest.raises(ValueError, match="уникальными"):
        ui.safe_read_csv(b"name,name\nA,B\n")
    state = {ui.RESULT_KEY: {"result": "old"}, ui.CONTEXT_KEY: "valid-file"}
    monkeypatch.setattr(ui.st, "session_state", state)
    ui.sync_context("invalid-file")
    assert ui.RESULT_KEY not in state
