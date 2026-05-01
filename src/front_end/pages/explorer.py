from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.back_end.config.settings import (
    CLASSIFICATION_MODEL_PATH,
    FEATURES_PARQUET_PATH,
    METER_ID_COL,
    TIMESTAMP_COL,
    VALUE_COL,
)
from src.back_end.training_pipelines.classification import (
    MODEL_REGISTRY,
    load_model,
    train_classifier,
)
from src.back_end.training_pipelines.regression import (
    apply_inhabited_threshold,
    load_model as load_reg_model,
    train_regressor,
)
from src.back_end.utils.data_loader import load_labels, load_raw_data
from src.back_end.utils.feature_engineering import build_features, get_feature_columns
# Cached loaders

@st.cache_data(show_spinner="Loading features…")
def _load_features() -> pd.DataFrame:
    if FEATURES_PARQUET_PATH.exists():
        return pd.read_parquet(FEATURES_PARQUET_PATH)
    raw    = load_raw_data()
    labels = load_labels()
    return build_features(raw, labels_df=labels)


@st.cache_data(show_spinner="Loading raw time series…")
def _load_raw() -> pd.DataFrame:
    return load_raw_data()


@st.cache_resource(show_spinner="Training classifier for explorer…")
def _get_clf_result(data_hash: str):
    feats = _load_features()
    return train_classifier(feats, model_name="Gradient Boosting")


@st.cache_resource(show_spinner="Training regressor for explorer…")
def _get_reg_result(data_hash: str):
    feats = _load_features()
    return train_regressor(feats)
# Page

st.title("Property Explorer")
st.markdown(
    "Select any meter to inspect its full-year consumption profile, "
    "model predictions, and key indicators."
)
st.info(
    "Pick any of the 500 properties from the sidebar to see its full consumption history, "
    "what the model predicts for it, and a breakdown of the features that drove that prediction. "
    "Use this page to spot unusual properties or verify the model's reasoning on specific cases."
)

try:
    features = _load_features()
except Exception as e:
    st.error(
        f"Failed to load feature data: **{e}**  \n\n"
        "Run `python scripts/precompute.py` from the project root to generate "
        "`data/processed/features.parquet`."
    )
    st.stop()
feat_cols = get_feature_columns(features)
data_hash = str(hash(tuple(features.columns.tolist())))
all_ids  = sorted(features[METER_ID_COL].astype(str).tolist())

# Sidebar search + select
st.sidebar.header("Property Selector")
search = st.sidebar.text_input("Search meter ID (partial match)")
filtered_ids = [i for i in all_ids if search.lower() in i.lower()] if search else all_ids
selected_id  = st.sidebar.selectbox("Meter ID", filtered_ids)

threshold = st.sidebar.slider(
    "Occupancy threshold",
    min_value=0.70, max_value=1.0, value=0.95, step=0.01,
)

if not selected_id:
    st.info("Select a meter ID from the sidebar.")
    st.stop()
prop_row = features[features[METER_ID_COL].astype(str) == selected_id]
if prop_row.empty:
    st.error(f"No features found for meter {selected_id}.")
    st.stop()

prop_row = prop_row.iloc[0]
true_label = "Secondary" if prop_row["residence_type"] == 1 else "Principal"
clf_result = _get_clf_result(data_hash)
reg_result = _get_reg_result(data_hash)

X_prop = pd.DataFrame([prop_row[feat_cols].fillna(0).values], columns=feat_cols)

# Classification
try:
    clf_model = clf_result.model
    proba     = clf_model.predict_proba(X_prop)[0, 1]
    pred_label = "Secondary" if proba >= 0.5 else "Principal"
except Exception:
    proba      = float("nan")
    pred_label = "N/A"

# Regression
try:
    reg_model   = reg_result.model
    occ_score   = float(reg_model.predict(X_prop)[0])
    is_inhabited = int(apply_inhabited_threshold([occ_score], threshold)[0])
except Exception:
    occ_score    = float("nan")
    is_inhabited = -1
st.subheader(f"Meter: {selected_id}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ground Truth",        true_label)
c2.metric("Predicted Type",      pred_label,
           delta=f"confidence {proba:.1%}" if not np.isnan(proba) else "")
c3.metric("Occupancy Score",     f"{occ_score:.2f}" if not np.isnan(occ_score) else "N/A")
c4.metric("Inhabited?",          "Yes" if is_inhabited == 1 else "No")
c5.metric("Cluster",             str(int(prop_row["cluster"])) if "cluster" in prop_row.index else "—")
st.caption(
    "Confidence shows how certain the model is (0–100 %). Above 70 % is reliable. "
    "Below 55 % means this property sits near the boundary between the two classes "
    "and the prediction should be read with care."
)
st.divider()
raw = _load_raw()
prop_raw = raw[raw[METER_ID_COL].astype(str) == selected_id].copy()
prop_raw[TIMESTAMP_COL] = pd.to_datetime(prop_raw[TIMESTAMP_COL], utc=True)
prop_raw["date"]  = prop_raw[TIMESTAMP_COL].dt.date
prop_raw["month"] = prop_raw[TIMESTAMP_COL].dt.month
prop_raw["hour"]  = prop_raw[TIMESTAMP_COL].dt.hour

if prop_raw.empty:
    st.warning("No raw time-series data found for this meter.")
else:
    tab_ts, tab_heatmap, tab_seasonal, tab_features = st.tabs([
        "Daily Time Series",
        "Hour-of-Day Heatmap",
        "Seasonal Profile",
        "Feature Summary",
    ])
# daily time series
    with tab_ts:
        daily = (
            prop_raw.groupby("date")[VALUE_COL]
            .sum()
            .reset_index()
            .rename(columns={"date": "Date", VALUE_COL: "Daily Consumption (Wh)"})
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date"],
            y=daily["Daily Consumption (Wh)"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#2196F3", width=1.5),
            name="Consumption",
        ))
        # Mark zero-consumption days
        zero_days = daily[daily["Daily Consumption (Wh)"] < 10]
        fig.add_trace(go.Scatter(
            x=zero_days["Date"],
            y=zero_days["Daily Consumption (Wh)"],
            mode="markers",
            marker=dict(color="#FF5722", size=6, symbol="x"),
            name="Zero / near-zero day",
        ))
        fig.update_layout(
            title=f"Daily Consumption — Meter {selected_id}",
            xaxis_title="Date",
            yaxis_title="Wh",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total (MWh)",    f"{daily['Daily Consumption (Wh)'].sum() / 1e6:.3f}")
        col2.metric("Daily Mean (Wh)", f"{daily['Daily Consumption (Wh)'].mean():.0f}")
        col3.metric("Zero days",       f"{(daily['Daily Consumption (Wh)'] < 10).sum()}")
        st.caption(
            "Red markers flag days with near-zero consumption (< 10 Wh). "
            "Long unbroken runs of red markers are the clearest sign of a secondary residence."
        )
# hour-of-day × month heatmap
    with tab_heatmap:
        pivot = (
            prop_raw.groupby(["month", "hour"])[VALUE_COL]
            .mean()
            .reset_index()
            .pivot(index="hour", columns="month", values=VALUE_COL)
        )
        month_names = {
            1: "Jan", 2: "Feb",  3: "Mar",  4: "Apr",
            5: "May", 6: "Jun",  7: "Jul",  8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        pivot.columns = [month_names.get(c, str(c)) for c in pivot.columns]

        fig = px.imshow(
            pivot,
            title=f"Mean Consumption (Wh) by Hour × Month — Meter {selected_id}",
            labels=dict(x="Month", y="Hour of Day", color="Wh"),
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Bright cells indicate high consumption.  "
            "A secondary residence will show bright colours only during vacation months."
        )
        st.markdown(
            "**Reading the heatmap:** A primary residence shows a fairly even bright pattern "
            "year-round with consistent evening hours lit up. A secondary residence shows dark "
            "columns (empty months) broken by bright patches during school holidays or summer."
        )
# seasonal profile
    with tab_seasonal:
        season_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring",  4: "Spring", 5: "Spring",
            6: "Summer",  7: "Summer", 8: "Summer",
            9: "Autumn",  10: "Autumn", 11: "Autumn",
        }
        prop_raw["season"] = prop_raw["month"].map(season_map)
        season_daily = (
            prop_raw.groupby(["season", "date"])[VALUE_COL]
            .sum()
            .reset_index()
            .groupby("season")[VALUE_COL]
            .mean()
            .reindex(["Winter", "Spring", "Summer", "Autumn"])
            .reset_index()
            .rename(columns={VALUE_COL: "Mean Daily Consumption (Wh)"})
        )
        fig = px.bar(
            season_daily,
            x="season", y="Mean Daily Consumption (Wh)",
            title=f"Mean Daily Consumption by Season — Meter {selected_id}",
            color="season",
            color_discrete_map={
                "Winter": "#1565C0", "Spring": "#388E3C",
                "Summer": "#F57F17", "Autumn": "#BF360C",
            },
        )
        st.plotly_chart(fig, width="stretch")

        # Weekday vs weekend
        prop_raw["is_weekend"] = prop_raw[TIMESTAMP_COL].dt.dayofweek.isin([5, 6])
        wk_comp = (
            prop_raw.groupby("is_weekend")[VALUE_COL]
            .mean()
            .reset_index()
            .assign(Day=lambda d: d["is_weekend"].map({True: "Weekend", False: "Weekday"}))
        )
        fig2 = px.bar(
            wk_comp, x="Day", y=VALUE_COL,
            title="Mean Half-Hour Consumption: Weekday vs. Weekend",
            color="Day",
            color_discrete_map={"Weekday": "#2196F3", "Weekend": "#FF5722"},
            labels={VALUE_COL: "Mean Consumption (Wh)"},
        )
        st.plotly_chart(fig2, width="stretch")
# feature summary
    with tab_features:
        key_features = [
            "mean_consumption", "std_consumption", "cv_consumption",
            "zero_days_ratio", "max_consecutive_zero_days",
            "winter_summer_ratio", "weekend_weekday_ratio",
            "peak_share", "night_baseline", "monthly_entropy",
            "active_weeks_ratio", "daily_autocorr",
            "summer_zero_ratio", "winter_zero_ratio",
        ]
        key_features = [f for f in key_features if f in prop_row.index]

        summary = pd.DataFrame({
            "Feature": key_features,
            "Value":   [round(float(prop_row[f]), 4) for f in key_features],
        })
        st.dataframe(summary, width="stretch")

        # Radar chart of normalised key features
        norm_feats = [
            "zero_days_ratio", "active_weeks_ratio", "weekend_weekday_ratio",
            "peak_share", "daily_autocorr", "monthly_entropy",
        ]
        norm_feats = [f for f in norm_feats if f in features.columns]
        if norm_feats:
            st.caption(
                "The radar axes are scaled so 0 = lowest value in the dataset, 1 = highest. "
                "A score of 0.5 means this property is exactly at the dataset median for that feature."
            )
            mins  = features[norm_feats].min()
            maxs  = features[norm_feats].max()
            norms = ((prop_row[norm_feats] - mins) / (maxs - mins + 1e-9)).clip(0, 1)

            fig = go.Figure(go.Scatterpolar(
                r=norms.values.tolist() + [norms.values[0]],
                theta=norm_feats + [norm_feats[0]],
                fill="toself",
                line=dict(color="#2196F3"),
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"Normalised Feature Radar — Meter {selected_id}",
            )
            st.plotly_chart(fig, width="stretch")
