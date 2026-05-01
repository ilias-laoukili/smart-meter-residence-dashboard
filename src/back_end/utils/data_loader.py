from __future__ import annotations

import pandas as pd

from src.back_end.config.settings import (
    LABELS_CSV_PATH,
    METER_ID_COL,
    RAW_CSV_PATH,
    TIMESTAMP_COL,
    VALUE_COL,
)


def load_raw_data(path: str | None = None, sample_frac: float | None = None) -> pd.DataFrame:
    """Load the raw half-hourly consumption CSV."""
    csv_path = path or str(RAW_CSV_PATH)
    df = pd.read_csv(csv_path)
    df[METER_ID_COL] = df[METER_ID_COL].astype(str).str.strip('"')
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42)

    return df


def load_labels(path: str | None = None) -> pd.DataFrame:
    """Load the professor's ground-truth label file (id, label, cluster)."""
    csv_path = path or str(LABELS_CSV_PATH)
    df = pd.read_csv(csv_path)
    df[METER_ID_COL] = df[METER_ID_COL].astype(str).str.strip('"')
    df["label"] = df["label"].astype(int)
    df["cluster"] = df["cluster"].astype(int)
    return df
