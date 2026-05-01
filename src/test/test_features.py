"""Tests for src/back_end/utils/feature_engineering.py."""
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.back_end.utils.feature_engineering import (
    _max_consecutive_zeros,
    build_features,
    get_feature_columns,
)

def _make_raw(n_days: int = 60, n_meters: int = 2, value_fn=None) -> pd.DataFrame:
    """Build a minimal raw DataFrame with n_meters × n_days × 48 readings."""
    records = []
    for meter in [f"M{i:03d}" for i in range(n_meters)]:
        for day_offset in range(n_days):
            base = datetime(2024, 1, 1, tzinfo=timezone.utc)
            date = base.replace(day=1) if False else base
            import pandas as _pd
            day_dt = _pd.Timestamp("2024-01-01", tz="UTC") + _pd.Timedelta(days=day_offset)
            for slot in range(48):
                ts = day_dt + _pd.Timedelta(minutes=30 * slot)
                val = value_fn(meter, day_offset, slot) if value_fn else 100.0
                records.append({"id": meter, "horodate": ts, "valeur": float(val)})
    return pd.DataFrame(records)


def _make_labels(meter_ids, label_map=None) -> pd.DataFrame:
    rows = []
    for i, mid in enumerate(meter_ids):
        label = label_map.get(mid, 0) if label_map else 0
        rows.append({"id": mid, "label": label, "cluster": i % 3})
    return pd.DataFrame(rows)

def test_build_features_shape():
    raw = _make_raw(n_days=30, n_meters=3)
    labels = _make_labels(["M000", "M001", "M002"])
    feats = build_features(raw, labels_df=labels)
    assert len(feats) == 3


def test_build_features_has_targets():
    raw = _make_raw(n_days=30, n_meters=2)
    labels = _make_labels(["M000", "M001"])
    feats = build_features(raw, labels_df=labels)
    assert "residence_type" in feats.columns
    assert "occupancy_rate" in feats.columns
    assert "is_inhabited" in feats.columns


def test_build_features_residence_type_from_labels():
    raw = _make_raw(n_days=30, n_meters=2)
    labels = _make_labels(["M000", "M001"], label_map={"M000": 0, "M001": 1})
    feats = build_features(raw, labels_df=labels)
    row_m001 = feats[feats["id"] == "M001"].iloc[0]
    assert row_m001["residence_type"] == 1


def test_build_features_occupancy_rate_bounded():
    raw = _make_raw(n_days=30, n_meters=2)
    labels = _make_labels(["M000", "M001"])
    feats = build_features(raw, labels_df=labels)
    assert feats["occupancy_rate"].between(0, 1).all()


def test_build_features_no_labels_fallback():
    raw = _make_raw(n_days=30, n_meters=3)
    feats = build_features(raw, labels_df=None)
    assert "residence_type" in feats.columns
    assert (feats["cluster"] == -1).all()


def test_build_features_unmatched_label_warns():
    """Meters not in labels file should produce a UserWarning."""
    raw = _make_raw(n_days=30, n_meters=2)
    labels = _make_labels(["M000"])  # M001 is missing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = build_features(raw, labels_df=labels)
    messages = [str(w.message) for w in caught]
    assert any("no matching label" in m or "meter(s) have no" in m for m in messages)


def test_build_features_unmatched_defaults_to_zero():
    raw = _make_raw(n_days=30, n_meters=2)
    labels = _make_labels(["M000"])  # M001 missing → should get residence_type=0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        feats = build_features(raw, labels_df=labels)
    row_m001 = feats[feats["id"] == "M001"].iloc[0]
    assert row_m001["residence_type"] == 0

def test_monthly_entropy_zero_consumption():
    """A property with zero consumption should have entropy=0, not max entropy."""
    raw = _make_raw(n_days=90, n_meters=1, value_fn=lambda m, d, s: 0.0)
    labels = _make_labels(["M000"])
    feats = build_features(raw, labels_df=labels)
    assert feats["monthly_entropy"].iloc[0] == 0.0


def test_monthly_entropy_positive_for_active_property():
    raw = _make_raw(n_days=90, n_meters=1, value_fn=lambda m, d, s: 200.0)
    labels = _make_labels(["M000"])
    feats = build_features(raw, labels_df=labels)
    assert feats["monthly_entropy"].iloc[0] > 0.0

def test_winter_summer_ratio_no_summer_data():
    """When there are no summer readings, ratio should not be NaN."""
    def only_winter_value(meter, day, slot):
        return 500.0

    # Use only January data (winter) → no summer months
    raw = _make_raw(n_days=31, n_meters=1, value_fn=only_winter_value)
    labels = _make_labels(["M000"])
    feats = build_features(raw, labels_df=labels)
    ratio = feats["winter_summer_ratio"].iloc[0]
    assert not np.isnan(ratio)
    assert ratio == 10.0  # hardcoded fallback when summer_mean == 0

def test_get_feature_columns_excludes_targets():
    raw = _make_raw(n_days=30, n_meters=2)
    labels = _make_labels(["M000", "M001"])
    feats = build_features(raw, labels_df=labels)
    feat_cols = get_feature_columns(feats)
    excluded = {"id", "residence_type", "occupancy_rate", "is_inhabited", "cluster"}
    assert not excluded.intersection(feat_cols)


def test_get_feature_columns_nonempty():
    raw = _make_raw(n_days=30, n_meters=2)
    labels = _make_labels(["M000", "M001"])
    feats = build_features(raw, labels_df=labels)
    assert len(get_feature_columns(feats)) > 10

def test_max_consecutive_zeros_all_zeros():
    s = pd.Series([0, 0, 0, 0, 0])
    assert _max_consecutive_zeros(s) == 5


def test_max_consecutive_zeros_no_zeros():
    s = pd.Series([100, 200, 50, 300])
    assert _max_consecutive_zeros(s) == 0


def test_max_consecutive_zeros_mixed():
    s = pd.Series([100, 0, 0, 50, 0, 0, 0, 200])
    assert _max_consecutive_zeros(s) == 3


def test_max_consecutive_zeros_single_element():
    # < 10 Wh counts as a zero day
    assert _max_consecutive_zeros(pd.Series([0])) == 1
    assert _max_consecutive_zeros(pd.Series([9])) == 1    # still < 10
    assert _max_consecutive_zeros(pd.Series([10])) == 0   # exactly 10 is not zero
