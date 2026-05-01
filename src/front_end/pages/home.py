from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.back_end.config.settings import FEATURES_PARQUET_PATH
from src.back_end.utils.data_loader import load_labels, load_raw_data
from src.back_end.utils.feature_engineering import build_features, get_feature_columns
# Cached data loading

@st.cache_data(show_spinner="Loading features…")
def _load_features() -> pd.DataFrame:
    """Load pre-computed features from Parquet, or recompute if not found."""
    if FEATURES_PARQUET_PATH.exists():
        return pd.read_parquet(FEATURES_PARQUET_PATH)
    raw    = load_raw_data()
    labels = load_labels()
    return build_features(raw, labels_df=labels)
# Page

st.title("Real Estate Analysis Dashboard")
st.markdown(
    "Analyse electricity smart-meter data (500 properties, ~8.7 M readings, "
    "Nov 2023 – Oct 2024) to classify **Principal vs. Secondary residences** "
    "and estimate **occupancy rates**."
)
st.info(
    "This dashboard analyses one full year of electricity consumption for 500 residential "
    "properties to answer two questions:\n\n"
    "1. **Is this property a primary or secondary residence?**  \n"
    "   Primary residences are lived in most of the year. Secondary residences (holiday homes, "
    "investment properties) are used seasonally and show long stretches of near-zero consumption.\n\n"
    "2. **Is the property currently occupied?**  \n"
    "   Based on how many days per year show meaningful electricity use.\n\n"
    "Use the left sidebar to navigate between pages."
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
n_total      = len(features)
n_secondary  = int(features["residence_type"].sum())
n_principal  = n_total - n_secondary
occ_mean     = features["occupancy_rate"].mean()
inhabited_n  = int(features["is_inhabited"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Properties",      f"{n_total}")
k2.metric("Principal Residences",  f"{n_principal}")
k3.metric("Secondary Residences",  f"{n_secondary}",
           delta=f"{n_secondary / n_total:.1%} of portfolio",
           delta_color="off")
k4.metric("Avg. Occupancy Rate",   f"{occ_mean:.1%}")
st.caption(
    "Avg. Occupancy Rate — fraction of days in the year with consumption above 500 Wh. "
    "A score of 0.95 means the property was active 95 % of days."
)
st.divider()
tab_dist, tab_seasonal, tab_cluster, tab_corr = st.tabs([
    "Distributions",
    "Seasonal Patterns",
    "Cluster Analysis",
    "Feature Correlations",
])
# tab 1: distributions
with tab_dist:
    st.markdown("How do primary and secondary properties differ in their consumption patterns?")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            names=["Principal", "Secondary"],
            values=[n_principal, n_secondary],
            title="Residence Type Breakdown",
            color_discrete_sequence=["#2196F3", "#FF5722"],
            hole=0.45,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.histogram(
            features,
            x="occupancy_rate",
            nbins=30,
            color=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            barmode="overlay",
            opacity=0.75,
            title="Occupancy Rate Distribution by Residence Type",
            labels={"occupancy_rate": "Occupancy Rate", "color": "Type"},
            color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
        )
        st.plotly_chart(fig, width="stretch")

    fig = px.box(
        features.melt(
            id_vars=["residence_type"],
            value_vars=["mean_night", "mean_morning", "mean_afternoon", "mean_evening"],
            var_name="Period",
            value_name="Consumption (Wh)",
        ).assign(
            Type=lambda d: d["residence_type"].map({0: "Principal", 1: "Secondary"}),
            Period=lambda d: d["Period"].str.replace("mean_", "").str.capitalize(),
        ),
        x="Period",
        y="Consumption (Wh)",
        color="Type",
        title="Consumption by Time Period and Residence Type",
        color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
    )
    st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(
            features,
            x="max_consecutive_zero_days",
            color=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            title="Max Consecutive Zero-Consumption Days",
            labels={"max_consecutive_zero_days": "Days", "color": "Type"},
            color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
        )
        st.plotly_chart(fig, width="stretch")

    with col4:
        fig = px.histogram(
            features,
            x="monthly_entropy",
            color=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            title="Monthly Consumption Entropy",
            labels={"monthly_entropy": "Entropy (nats)", "color": "Type"},
            color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
        )
        st.plotly_chart(fig, width="stretch")
# tab 2: seasonal patterns
with tab_seasonal:
    st.markdown("Winter vs summer and weekday vs weekend — key signals for identifying holiday homes.")
    fig = px.scatter(
        features,
        x="winter_mean",
        y="summer_mean",
        color=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
        title="Winter vs. Summer Mean Consumption",
        labels={
            "winter_mean":  "Winter Mean (Wh)",
            "summer_mean":  "Summer Mean (Wh)",
            "color":        "Type",
        },
        opacity=0.65,
        color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
    )
    max_val = max(features["winter_mean"].max(), features["summer_mean"].max())
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(dash="dash", color="grey"),
    )
    fig.add_annotation(
        x=max_val * 0.7, y=max_val * 0.75,
        text="winter = summer", showarrow=False, font=dict(color="grey"),
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            features,
            x=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            y="summer_zero_ratio",
            color=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            title="Summer Zero-Consumption Days Ratio",
            labels={"x": "Type", "summer_zero_ratio": "Ratio"},
            color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.box(
            features,
            x=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            y="winter_zero_ratio",
            color=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            title="Winter Zero-Consumption Days Ratio",
            labels={"x": "Type", "winter_zero_ratio": "Ratio"},
            color_discrete_map={"Principal": "#2196F3", "Secondary": "#FF5722"},
        )
        st.plotly_chart(fig, width="stretch")

    grp = (
        features.groupby(
            features["residence_type"].map({0: "Principal", 1: "Secondary"})
        )[["weekday_mean", "weekend_mean"]]
        .mean()
        .reset_index()
        .rename(columns={"residence_type": "Type"})
    )
    fig = px.bar(
        grp,
        x="Type",
        y=["weekday_mean", "weekend_mean"],
        barmode="group",
        title="Weekday vs. Weekend Mean Consumption",
        labels={"value": "Mean Consumption (Wh)", "variable": "Day type"},
    )
    st.plotly_chart(fig, width="stretch")
# tab 3: cluster analysis
with tab_cluster:
    st.markdown(
        "The professor's pre-assigned clusters group properties by consumption profile. "
        "This tab checks how well those clusters align with residence type and occupancy. "
        "Clusters with a high secondary rate (red bars) are dominated by holiday homes."
    )

    if "cluster" in features.columns and features["cluster"].nunique() > 1:
        col1, col2 = st.columns(2)

        with col1:
            cluster_type = (
                features.groupby("cluster")["residence_type"]
                .mean()
                .reset_index()
                .rename(columns={"residence_type": "secondary_rate"})
            )
            fig = px.bar(
                cluster_type.sort_values("cluster"),
                x="cluster",
                y="secondary_rate",
                title="Secondary Residence Rate by Cluster",
                labels={"cluster": "Cluster", "secondary_rate": "Secondary rate"},
                color="secondary_rate",
                color_continuous_scale="Reds",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        with col2:
            cluster_occ = (
                features.groupby("cluster")["occupancy_rate"]
                .mean()
                .reset_index()
            )
            fig = px.bar(
                cluster_occ.sort_values("cluster"),
                x="cluster",
                y="occupancy_rate",
                title="Average Occupancy Rate by Cluster",
                labels={"cluster": "Cluster", "occupancy_rate": "Occupancy rate"},
                color="occupancy_rate",
                color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        cluster_stats = (
            features.groupby("cluster")
            .agg(
                n_properties=("id", "count"),
                secondary_rate=("residence_type", "mean"),
                occupancy_rate=("occupancy_rate", "mean"),
                mean_consumption=("mean_consumption", "mean"),
                zero_days_ratio=("zero_days_ratio", "mean"),
            )
            .round(3)
            .reset_index()
        )
        st.dataframe(cluster_stats, width="stretch")

        # Scatter: winter vs summer coloured by cluster
        fig = px.scatter(
            features,
            x="winter_mean",
            y="summer_mean",
            color=features["cluster"].astype(str),
            symbol=features["residence_type"].map({0: "Principal", 1: "Secondary"}),
            title="Consumption Profile by Cluster and Residence Type",
            labels={
                "winter_mean": "Winter Mean (Wh)",
                "summer_mean": "Summer Mean (Wh)",
                "color": "Cluster",
                "symbol": "Type",
            },
            opacity=0.7,
        )
        st.plotly_chart(fig, width="stretch")

    else:
        st.info(
            "Cluster metadata not available.  "
            "Run `scripts/precompute.py` to regenerate features with labels."
        )
# tab 4: feature correlations
with tab_corr:
    st.markdown(
        "Features that are strongly correlated (dark red) measure the same underlying signal. "
        "This matters when reading feature importance on the Modelling pages."
    )
    feat_cols = get_feature_columns(features)
    corr = features[feat_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".1f",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_layout(height=750)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Values near +1 or −1 = strong relationship. Near 0 = no relationship. "
        "Features that are dark red with each other carry redundant information."
    )
with st.expander("Glossary — what do these feature names mean?"):
    st.markdown("""
| Feature | Plain-English meaning |
|---|---|
| `cv_consumption` | How erratic daily usage is — high means big swings between active and idle days |
| `night_baseline` | Typical overnight draw — captures always-on appliances (fridge, boiler standby) |
| `peak_share` | Share of daily energy used during the evening peak (17–21 h) |
| `monthly_entropy` | How evenly spread consumption is across months — low entropy = concentrated in a few months |
| `max_consecutive_zero_days` | Longest unbroken absence streak in the year |
| `daily_autocorr` | How predictable tomorrow's usage is from today's — high = regular routine |
| `active_weeks_ratio` | Fraction of weeks with at least some measurable consumption |
| `winter_summer_ratio` | Winter consumption divided by summer — high ratio typical of year-round heated homes |
""")
with st.expander("Preview engineered feature table (first 50 rows)"):
    display_cols = ["id", "residence_type", "cluster", "occupancy_rate",
                    "is_inhabited"] + feat_cols[:10]
    display_cols = [c for c in display_cols if c in features.columns]
    st.dataframe(features[display_cols].head(50), width="stretch")
