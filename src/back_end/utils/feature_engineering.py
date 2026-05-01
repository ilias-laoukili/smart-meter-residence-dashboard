# Feature engineering: aggregate half-hourly smart-meter readings into
# one row per property for classification and regression.

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from src.back_end.config.settings import (
    METER_ID_COL,
    OCCUPANCY_DAY_THRESHOLD_WH,
    READINGS_PER_DAY,
    SUMMER_MONTHS,
    TIMESTAMP_COL,
    VALUE_COL,
    WINTER_MONTHS,
)

_NIGHT     = range(0,  6)
_MORNING   = range(6,  12)
_AFTERNOON = range(12, 18)
_EVENING   = range(18, 24)


def _period_label(hour: int) -> str:
    if hour in _NIGHT:
        return "night"
    if hour in _MORNING:
        return "morning"
    if hour in _AFTERNOON:
        return "afternoon"
    return "evening"


def build_features(df: pd.DataFrame, labels_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate raw readings into one row per property with engineered features."""
    df = df.copy()
    df["hour"] = df[TIMESTAMP_COL].dt.hour
    df["month"] = df[TIMESTAMP_COL].dt.month
    df["dayofweek"] = df[TIMESTAMP_COL].dt.dayofweek   # 0 = Monday
    df["date"] = df[TIMESTAMP_COL].dt.date
    df["week"] = df[TIMESTAMP_COL].dt.isocalendar().week.astype(int)
    df["period"] = df["hour"].apply(_period_label)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_winter"] = df["month"].isin(WINTER_MONTHS).astype(int)
    df["is_summer"] = df["month"].isin(SUMMER_MONTHS).astype(int)

    features = (
        df.groupby(METER_ID_COL)
        .apply(_agg_property, include_groups=False)
        .reset_index()
    )

    if labels_df is not None:
        lbl = labels_df[[METER_ID_COL, "label", "cluster"]].copy()
        lbl[METER_ID_COL] = lbl[METER_ID_COL].astype(str)
        features[METER_ID_COL] = features[METER_ID_COL].astype(str)
        features = features.merge(lbl, on=METER_ID_COL, how="left")
        unmatched = features["label"].isna().sum()
        if unmatched > 0:
            warnings.warn(
                f"{unmatched} meter(s) have no matching label in the labels file "
                "and will be assigned residence_type=0 (Principal). "
                "Check that the labels CSV covers all meter IDs in the raw data.",
                stacklevel=2,
            )
        features["residence_type"] = features["label"].fillna(0).astype(int)
        features.drop(columns=["label"], inplace=True)
    else:
        features = _derive_heuristic_residence_type(features)
        features["cluster"] = -1

    features = _derive_occupancy(features)
    return features


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the training feature column names (excludes id, targets, metadata)."""
    exclude = {
        METER_ID_COL,
        "residence_type",
        "occupancy_rate",
        "is_inhabited",
        "cluster",
    }
    return [c for c in features_df.columns if c not in exclude]


def _agg_property(group: pd.DataFrame) -> pd.Series:
    """Compute all features for a single property."""
    val = group[VALUE_COL]
    daily_tot = group.groupby("date")[VALUE_COL].sum()
    n_days = len(daily_tot)
    feats: dict[str, float] = {}

    # global stats
    feats["mean_consumption"] = val.mean()
    feats["std_consumption"] = val.std()
    feats["median_consumption"] = val.median()
    feats["max_consumption"] = val.max()
    feats["cv_consumption"] = (
        val.std() / val.mean() if val.mean() > 0 else 0.0
    )
    feats["total_consumption"] = val.sum()

    # time-of-day periods
    for period in ("night", "morning", "afternoon", "evening"):
        mask = group["period"] == period
        feats[f"mean_{period}"] = val[mask].mean() if mask.any() else 0.0

    # night baseline: median is more robust to spikes than mean
    night_mask = group["hour"].isin(_NIGHT)
    feats["night_baseline"] = (
        float(val[night_mask].median()) if night_mask.any() else 0.0
    )

    # seasonal
    w_mask = group["is_winter"] == 1
    s_mask = group["is_summer"] == 1
    winter_mean = val[w_mask].mean() if w_mask.any() else 0.0
    summer_mean = val[s_mask].mean() if s_mask.any() else 0.0
    feats["winter_mean"] = winter_mean
    feats["summer_mean"] = summer_mean
    feats["winter_summer_ratio"] = (
        winter_mean / summer_mean if summer_mean > 0 else 10.0
    )

    # weekday vs weekend
    wkday = val[group["is_weekend"] == 0]
    wkend = val[group["is_weekend"] == 1]
    wkday_mean = wkday.mean() if len(wkday) > 0 else 0.0
    wkend_mean = wkend.mean() if len(wkend) > 0 else 0.0
    feats["weekday_mean"] = wkday_mean
    feats["weekend_mean"] = wkend_mean
    feats["weekend_weekday_ratio"] = (
        wkend_mean / wkday_mean if wkday_mean > 0 else 1.0
    )

    # zero / low consumption days
    feats["n_days_recorded"] = n_days
    feats["zero_days_ratio"] = (
        (daily_tot < 10).sum() / n_days if n_days > 0 else 0.0
    )
    feats["low_consumption_days_ratio"] = (
        (daily_tot < daily_tot.quantile(0.1)).sum() / n_days
        if n_days > 1 else 0.0
    )

    # season-specific zero days
    summer_dates = set(group.loc[group["is_summer"] == 1, "date"].unique())
    winter_dates = set(group.loc[group["is_winter"] == 1, "date"].unique())
    summer_daily = daily_tot[daily_tot.index.isin(summer_dates)]
    winter_daily = daily_tot[daily_tot.index.isin(winter_dates)]
    feats["summer_zero_ratio"] = (
        (summer_daily < 10).sum() / len(summer_daily)
        if len(summer_daily) > 0 else 0.0
    )
    feats["winter_zero_ratio"] = (
        (winter_daily < 10).sum() / len(winter_daily)
        if len(winter_daily) > 0 else 0.0
    )

    # peak-hour share (17–21h evening peak)
    peak_mask = group["hour"].between(17, 21)
    feats["peak_share"] = (
        val[peak_mask].sum() / val.sum() if val.sum() > 0 else 0.0
    )

    # lag-1 autocorrelation of daily totals
    feats["daily_autocorr"] = (
        daily_tot.autocorr(lag=1) if len(daily_tot) > 2 else 0.0
    )

    # longest streak of near-zero days
    feats["max_consecutive_zero_days"] = _max_consecutive_zeros(daily_tot)

    # monthly entropy: low entropy = consumption concentrated in a few months (secondary)
    monthly = group.groupby("month")[VALUE_COL].sum()
    total_consumption = monthly.sum()
    if total_consumption > 0:
        monthly_share = monthly / total_consumption
        feats["monthly_entropy"] = float(scipy_entropy(monthly_share + 1e-9))
    else:
        feats["monthly_entropy"] = 0.0

    # fraction of weeks with at least some consumption
    week_active = group.groupby("week")[VALUE_COL].sum() > 10
    feats["active_weeks_ratio"] = (
        week_active.sum() / len(week_active) if len(week_active) > 0 else 0.0
    )

    # occupancy rate (regression target)
    feats["occupancy_rate"] = (
        (daily_tot > OCCUPANCY_DAY_THRESHOLD_WH).sum() / n_days
        if n_days > 0 else 0.0
    )

    return pd.Series(feats)


def _max_consecutive_zeros(daily_series: pd.Series) -> int:
    """Return the longest streak of consecutive near-zero days."""
    is_zero = (daily_series < 10).astype(int).values
    max_streak = cur_streak = 0
    for v in is_zero:
        if v:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    return int(max_streak)


def _derive_occupancy(features: pd.DataFrame) -> pd.DataFrame:
    """Add binary is_inhabited flag from occupancy_rate."""
    features["is_inhabited"] = (features["occupancy_rate"] >= 0.95).astype(int)
    return features


def _derive_heuristic_residence_type(features: pd.DataFrame) -> pd.DataFrame:
    """Fallback heuristic label when no ground-truth file is available."""
    score = (
        features["zero_days_ratio"] * 3
        + features["low_consumption_days_ratio"] * 2
        + (1 / (features["winter_summer_ratio"] + 0.01)).clip(0, 2)
        + (
            features["mean_consumption"]
            < features["mean_consumption"].quantile(0.25)
        ).astype(float)
    )
    threshold = score.quantile(0.86)
    features["residence_type"] = (score >= threshold).astype(int)
    return features
