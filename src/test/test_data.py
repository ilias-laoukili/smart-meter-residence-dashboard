"""Tests for src/back_end/utils/data_loader.py."""
import io
import textwrap

import pandas as pd
import pytest

from src.back_end.utils.data_loader import load_labels, load_raw_data


RAW_CSV = textwrap.dedent("""\
    id,horodate,valeur
    "001",2024-01-01 00:00:00+00:00,100.0
    "001",2024-01-01 00:30:00+00:00,120.5
    "002",2024-01-01 00:00:00+00:00,0.0
""")

LABELS_CSV = textwrap.dedent("""\
    id,label,cluster
    "001",0,2
    "002",1,5
""")


@pytest.fixture()
def raw_csv_path(tmp_path):
    p = tmp_path / "export.csv"
    p.write_text(RAW_CSV)
    return str(p)


@pytest.fixture()
def labels_csv_path(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(LABELS_CSV)
    return str(p)


def test_load_raw_data_columns(raw_csv_path):
    df = load_raw_data(path=raw_csv_path)
    assert set(df.columns) == {"id", "horodate", "valeur"}


def test_load_raw_data_types(raw_csv_path):
    df = load_raw_data(path=raw_csv_path)
    assert pd.api.types.is_string_dtype(df["id"]) or df["id"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(df["horodate"])
    assert pd.api.types.is_float_dtype(df["valeur"])


def test_load_raw_data_strips_quotes(raw_csv_path):
    df = load_raw_data(path=raw_csv_path)
    assert '"' not in df["id"].iloc[0]


def test_load_raw_data_utc_aware(raw_csv_path):
    df = load_raw_data(path=raw_csv_path)
    assert df["horodate"].dt.tz is not None


def test_load_raw_data_sample_frac(raw_csv_path):
    df_full = load_raw_data(path=raw_csv_path)
    df_sampled = load_raw_data(path=raw_csv_path, sample_frac=0.5)
    assert len(df_sampled) < len(df_full)


def test_load_labels_columns(labels_csv_path):
    df = load_labels(path=labels_csv_path)
    assert set(df.columns) == {"id", "label", "cluster"}


def test_load_labels_types(labels_csv_path):
    df = load_labels(path=labels_csv_path)
    assert df["label"].dtype == int
    assert df["cluster"].dtype == int


def test_load_labels_strips_quotes(labels_csv_path):
    df = load_labels(path=labels_csv_path)
    assert '"' not in df["id"].iloc[0]


def test_load_labels_values(labels_csv_path):
    df = load_labels(path=labels_csv_path)
    assert set(df["label"].unique()).issubset({0, 1})
