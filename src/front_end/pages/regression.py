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
    FEATURES_PARQUET_PATH,
    OCCUPANCY_DAY_THRESHOLD_WH,
    REGRESSION_MODEL_PATH,
)
from src.back_end.training_pipelines.regression import (
    RegressionResult,
    apply_inhabited_threshold,
    evaluate_loaded_model,
    load_model,
    predict_full_dataset,
    save_model,
    train_regressor,
)
from src.back_end.utils.data_loader import load_labels, load_raw_data
from src.back_end.utils.feature_engineering import build_features, get_feature_columns
# Cached helpers

@st.cache_data(show_spinner="Loading features…")
def _load_features() -> pd.DataFrame:
    if FEATURES_PARQUET_PATH.exists():
        return pd.read_parquet(FEATURES_PARQUET_PATH)
    raw    = load_raw_data()
    labels = load_labels()
    return build_features(raw, labels_df=labels)


@st.cache_resource(show_spinner="Training regressor…")
def _train(data_hash: str) -> RegressionResult:
    feats = _load_features()
    return train_regressor(feats)
# Page

st.title("Regression — Occupancy Rate & Inhabited Status")
st.markdown(
    "The **occupancy rate** is a continuous score in [0, 1] derived from "
    "smart-meter consumption patterns (fraction of active days, weekly "
    "regularity, etc.).  No ground-truth labels exist for this target — it is "
    "a data-driven proxy.  \n\n"
    "A configurable **threshold** converts the continuous score into a binary "
    "**inhabited / not-inhabited** flag."
)
st.info(
    "The model predicts a score between 0 and 1 for each property, where 1 means the property "
    "showed electricity use on almost every day of the year. Properties above the threshold "
    "are labelled **inhabited**; those below are flagged as likely unoccupied for extended periods. "
    "Adjust the threshold slider in the sidebar to see how it changes the count."
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
st.sidebar.header("Occupancy Settings")
threshold = st.sidebar.slider(
    "Inhabited threshold",
    min_value=0.70, max_value=1.0, value=0.95, step=0.01,
    help=(
        "Properties with predicted occupancy ≥ threshold are labelled 'inhabited'.  "
        "The occupancy score is the fraction of days above the minimum consumption "
        f"threshold ({OCCUPANCY_DAY_THRESHOLD_WH} Wh).  In this dataset almost all "
        "meters are active most days, so a high threshold (0.90–0.98) is needed "
        "to discriminate genuinely absent properties."
    ),
)
action = st.radio("Model action", ["Train new model", "Load saved model"], horizontal=True)

result: RegressionResult | None = None

if action == "Train new model":
    if st.button("Train", type="primary"):
        result = _train(data_hash)
        save_model(result)
        st.success("Regressor trained and saved.")
else:
    if REGRESSION_MODEL_PATH.exists():
        result = evaluate_loaded_model(features)
        st.info("Pre-trained model loaded from disk and evaluated on the held-out test set.")
    else:
        st.warning("No saved model found. Train a model first.")

if result is None:
    st.info("Click **Train** or load a model to see results.")
    st.stop()
m1, m2, m3 = st.columns(3)
m1.metric("MAE",      f"{result.mae:.4f}")
m2.metric("RMSE",     f"{result.rmse:.4f}")
m3.metric("R² Score", f"{result.r2:.4f}")

with st.expander("What do these numbers mean?"):
    st.markdown("""
- **MAE (Mean Absolute Error)** — on average, the model's occupancy prediction is off by this
  many percentage points. An MAE of 0.04 means ±4 pp on average.
- **RMSE** — similar to MAE but larger errors count more. If RMSE is much bigger than MAE,
  a few properties are being predicted very poorly.
- **R² Score** — how much of the variation in occupancy the model explains. 0.90 means 90 %
  of the differences between properties are captured by the model.
""")

# Inhabited breakdown on full dataset
full_preds = predict_full_dataset(result, features)
full_preds["is_inhabited_pred"] = apply_inhabited_threshold(
    full_preds["occupancy_rate_pred"], threshold
)
n_inhabited    = int(full_preds["is_inhabited_pred"].sum())
n_uninhabited  = len(full_preds) - n_inhabited

st.markdown("---")
st.subheader(f"Inhabited Status at threshold = {threshold:.2f}")
i1, i2, i3 = st.columns(3)
i1.metric("Inhabited",     f"{n_inhabited}",
           delta=f"{n_inhabited / len(full_preds):.1%}")
i2.metric("Not Inhabited", f"{n_uninhabited}",
           delta=f"{n_uninhabited / len(full_preds):.1%}", delta_color="inverse")
i3.metric("Threshold",     f"{threshold:.2f}")
st.caption(
    "A property is counted as 'inhabited' when its predicted occupancy score is at or above "
    "the threshold. Adjust the slider in the sidebar to see how this changes the count."
)
st.divider()
tab_pred, tab_res, tab_fi, tab_breakdown = st.tabs([
    "Predicted vs. Actual",
    "Residuals",
    "Feature Importance",
    "Occupancy Breakdown",
])

# Predicted vs actual
with tab_pred:
    st.caption(
        "Each dot is one property. Dots on the diagonal line are perfect predictions. "
        "Dots above the line mean the model over-predicted occupancy; "
        "below means it under-predicted."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.y_test.values,
        y=result.y_pred,
        mode="markers",
        marker=dict(opacity=0.6, color="#2196F3"),
        name="Predictions",
    ))
    rng = [
        min(result.y_test.min(), result.y_pred.min()),
        max(result.y_test.max(), result.y_pred.max()),
    ]
    fig.add_trace(go.Scatter(
        x=rng, y=rng, mode="lines",
        line=dict(dash="dash", color="grey"), name="Perfect fit",
    ))
    fig.update_layout(
        title="Predicted vs. Actual Occupancy Rate (test set)",
        xaxis_title="Actual",
        yaxis_title="Predicted",
    )
    st.plotly_chart(fig, width="stretch")

# Residuals
with tab_res:
    st.caption(
        "A residual is the gap between what the model predicted and the actual occupancy score. "
        "A healthy model has residuals clustered near zero with no obvious pattern. "
        "A funnel shape suggests the model is less reliable at the extremes."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            x=result.residuals, nbins=30,
            title="Residual Distribution",
            labels={"x": "Residual (Actual − Predicted)"},
            color_discrete_sequence=["#2196F3"],
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.y_pred, y=result.residuals,
            mode="markers", marker=dict(opacity=0.6, color="#FF5722"),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(
            title="Residuals vs. Predicted",
            xaxis_title="Predicted Occupancy Rate",
            yaxis_title="Residual",
        )
        st.plotly_chart(fig, width="stretch")

# Feature importance
with tab_fi:
    top_n = st.slider(
        "Top N features", 5, len(feat_cols), min(15, len(feat_cols)), key="reg_fi"
    )
    imp = result.feature_importance.head(top_n)
    fig = px.bar(
        imp, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} Feature Importances",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")

# Occupancy breakdown
with tab_breakdown:
    col1, col2 = st.columns(2)

    with col1:
        # By residence type
        if "residence_type" in full_preds.columns:
            fig = px.box(
                full_preds,
                x=full_preds["residence_type"].map({0: "Principal", 1: "Secondary"}),
                y="occupancy_rate_pred",
                color=full_preds["residence_type"].map({0: "Principal", 1: "Secondary"}),
                title="Predicted Occupancy Rate by Residence Type",
                labels={"x": "Type", "occupancy_rate_pred": "Predicted Occupancy"},
                color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
            )
            fig.add_hline(y=threshold, line_dash="dash", line_color="green",
                          annotation_text=f"Threshold = {threshold:.2f}")
            st.plotly_chart(fig, width="stretch")

    with col2:
        # Distribution of predicted scores
        fig = px.histogram(
            full_preds,
            x="occupancy_rate_pred",
            nbins=30,
            title="Distribution of Predicted Occupancy Scores",
            labels={"occupancy_rate_pred": "Predicted Occupancy Rate"},
            color_discrete_sequence=["#2196F3"],
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="green",
                      annotation_text=f"Threshold = {threshold:.2f}")
        st.plotly_chart(fig, width="stretch")

    # Cluster breakdown
    if "cluster" in full_preds.columns and full_preds["cluster"].nunique() > 1:
        cluster_occ = (
            full_preds.groupby("cluster")
            .agg(
                mean_predicted_occupancy=("occupancy_rate_pred", "mean"),
                inhabited_rate=("is_inhabited_pred", "mean"),
                n=("occupancy_rate_pred", "count"),
            )
            .round(3)
            .reset_index()
        )
        fig = px.bar(
            cluster_occ.sort_values("cluster"),
            x="cluster",
            y=["mean_predicted_occupancy", "inhabited_rate"],
            barmode="group",
            title="Predicted Occupancy and Inhabited Rate by Cluster",
            labels={"value": "Rate", "cluster": "Cluster", "variable": "Metric"},
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(cluster_occ, width="stretch")

    # Full predictions table
    with st.expander("Full predicted occupancy table"):
        st.dataframe(
            full_preds.sort_values("occupancy_rate_pred").assign(
                inhabited=lambda d: d["is_inhabited_pred"].map({1: "Yes", 0: "No"}),
            ),
            width="stretch",
        )
