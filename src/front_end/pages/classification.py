from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.back_end.config.settings import (
    CLASSIFICATION_MODEL_PATH,
    FEATURES_PARQUET_PATH,
)
from src.back_end.training_pipelines.classification import (
    ClassificationResult,
    MODEL_REGISTRY,
    compare_models,
    evaluate_loaded_model,
    load_model,
    save_model,
    train_classifier,
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


@st.cache_resource(show_spinner="Training classifier…")
def _train(model_name: str, data_hash: str) -> ClassificationResult:
    feats = _load_features()
    return train_classifier(feats, model_name=model_name)


@st.cache_data(show_spinner="Running model comparison…")
def _compare(data_hash: str) -> pd.DataFrame:
    feats = _load_features()
    return compare_models(feats)
# Page

st.title("Classification — Principal vs. Secondary Residence")
st.markdown(
    "Ground-truth labels from the professor's annotation file "
    "(`RES2-6-9-labels.csv`).  "
    "**428** principal · **72** secondary (14.4 % positive rate).  "
    "Class imbalance is handled via sample-weight balancing for tree models."
)
st.info(
    "**What we're predicting:** whether a property is a *primary residence* (someone lives there "
    "most of the year) or a *secondary residence* (used occasionally — think holiday home or "
    "investment flat). The model learns to spot the difference from patterns like long "
    "zero-consumption stretches, very low winter use, or activity only during school holidays."
)
st.divider()

try:
    features = _load_features()
except Exception as e:
    st.error(
        f"Failed to load feature data: **{e}**  \n\n"
        "Run `python scripts/precompute.py` from the project root to generate "
        "`data/processed/features.parquet`."
    )
    st.stop()
feat_cols  = get_feature_columns(features)
data_hash  = str(hash(tuple(features.columns.tolist())))
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Algorithm", list(MODEL_REGISTRY.keys()), index=0
)
col_action, col_compare = st.columns([1, 1])

with col_action:
    action = st.radio(
        "Model action", ["Train new model", "Load saved model"], horizontal=True
    )

result: ClassificationResult | None = None

if action == "Train new model":
    if st.button("Train", type="primary"):
        result = _train(model_name, data_hash)
        save_model(result)
        st.success(f"{model_name} trained and saved.")
else:
    if CLASSIFICATION_MODEL_PATH.exists():
        result = evaluate_loaded_model(features)
        st.info("Pre-trained model loaded from disk and evaluated on the held-out test set.")
    else:
        st.warning("No saved model found. Train a model first.")

with col_compare:
    if st.button("Compare all models"):
        with st.spinner("Training 3 models…"):
            cmp_df = _compare(data_hash)
        st.dataframe(
            cmp_df.style.highlight_max(
                subset=["Accuracy", "F1", "ROC-AUC"], color="#d4edda"
            ),
            width="stretch",
        )

if result is None:
    st.info("Click **Train** or load a model to see results.")
    st.stop()
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{result.accuracy:.2%}")
m2.metric("F1 Score", f"{result.f1:.2%}")
m3.metric("ROC-AUC",  f"{result.roc_auc:.2%}")

with st.expander("What do these numbers mean?"):
    st.markdown("""
- **Accuracy** — overall % of properties correctly labelled. Can look good even on a bad model
  if one class is rare (here, only 14 % of properties are secondary residences).
- **F1 Score** — balances catching secondary residences (recall) against avoiding false alarms
  (precision). The number to watch here.
- **ROC-AUC** — how well the model separates the two classes regardless of the decision threshold.
  0.5 = no better than a coin flip; 1.0 = perfect. Above 0.85 is strong for this dataset.
""")

st.markdown("---")
tab_cm, tab_roc, tab_fi, tab_shap_global, tab_shap_local = st.tabs([
    "Confusion Matrix",
    "ROC Curve",
    "Feature Importance",
    "SHAP — Global",
    "SHAP — Per Property",
])

# Confusion matrix
with tab_cm:
    st.caption(
        "Rows = actual label, columns = predicted label. "
        "The top-right cell is properties the model wrongly labelled as secondary (false alarms). "
        "The bottom-left is secondary residences the model missed — the more costly error."
    )
    labels_cm = ["Principal", "Secondary"]
    fig = ff.create_annotated_heatmap(
        result.confusion,
        x=labels_cm,
        y=labels_cm,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title=f"Confusion Matrix — {result.model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.text(f"Classification Report:\n{result.report}")

# ROC curve
with tab_roc:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.fpr, y=result.tpr, mode="lines",
        name=f"AUC = {result.roc_auc:.3f}",
        line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="grey"), name="Random",
    ))
    fig.update_layout(
        title=f"ROC Curve — {result.model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The curve shows the trade-off between catching more secondary residences (higher recall) "
        "and generating more false alarms. A curve that hugs the top-left corner is better. "
        "The shaded area under the curve (AUC) summarises this in one number."
    )

# Feature importance
with tab_fi:
    top_n = st.slider("Top N features", 5, len(feat_cols), min(15, len(feat_cols)))
    imp   = result.feature_importance.head(top_n)
    fig   = px.bar(
        imp, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} Feature Importances — {result.model_name}",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")

# SHAP global beeswarm
with tab_shap_global:
    if result.shap_values is None:
        st.info(
            "SHAP values are available for tree-based models only "
            "(Gradient Boosting and Random Forest)."
        )
    else:
        st.info(
            "Each dot is one property from the test set. Its horizontal position shows how much "
            "that feature pushed the model's prediction. **Right of centre** = pushed toward "
            "*secondary residence*. **Left of centre** = pushed toward *primary residence*. "
            "Dot colour shows the raw feature value (red = high, blue = low)."
        )
        shap_df = pd.DataFrame(
            result.shap_values,
            columns=feat_cols,
        )
        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        top_feats = mean_abs.head(15).index.tolist()

        rows = []
        for feat in top_feats:
            vals = result.X_test[feat].values
            sv   = shap_df[feat].values
            for v, s in zip(vals, sv):
                rows.append({"feature": feat, "shap_value": s, "feature_value": v})
        bee_df = pd.DataFrame(rows)

        # Add jitter on y so dots spread like a beeswarm
        rng = np.random.default_rng(42)
        bee_df["y_jitter"] = bee_df["feature"].map(
            {f: i for i, f in enumerate(top_feats)}
        ) + rng.uniform(-0.3, 0.3, size=len(bee_df))

        fig = px.scatter(
            bee_df,
            x="shap_value",
            y="y_jitter",
            color="feature_value",
            color_continuous_scale="RdBu_r",
            title="SHAP Beeswarm — Global Feature Impact",
            labels={
                "shap_value":    "SHAP value (impact on prediction)",
                "feature_value": "Feature value",
            },
            opacity=0.6,
        )
        fig.update_layout(
            height=500,
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(top_feats))),
                ticktext=top_feats,
                title="Feature",
            ),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, width="stretch")

        # Mean |SHAP| bar
        mean_shap_df = mean_abs.head(15).reset_index()
        mean_shap_df.columns = ["feature", "mean_abs_shap"]
        fig2 = px.bar(
            mean_shap_df,
            x="mean_abs_shap", y="feature",
            orientation="h",
            title="Mean |SHAP| — Top 15 Features",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
        )
        fig2.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, width="stretch")

# SHAP per-property waterfall
with tab_shap_local:
    if result.shap_values is None:
        st.info(
            "SHAP values are available for tree-based models only "
            "(Gradient Boosting and Random Forest)."
        )
    else:
        test_indices = result.X_test.index.tolist()
        selected_idx = st.selectbox(
            "Select a test-set property (row index)",
            test_indices,
        )

        if selected_idx is not None:
            pos   = test_indices.index(selected_idx)
            sv    = result.shap_values[pos]
            sample = result.X_test.loc[selected_idx]
            pred_proba = result.y_proba[pos]
            pred_label = "Secondary" if pred_proba >= 0.5 else "Principal"
            true_label = "Secondary" if result.y_test.loc[selected_idx] == 1 else "Principal"

            st.markdown(
                f"**Predicted:** {pred_label} (probability = {pred_proba:.1%})  |  "
                f"**Actual:** {true_label}"
            )
            st.caption(
                "The bar chart builds up the model's prediction step by step, starting from the "
                "average property. Each bar adds or subtracts one feature's contribution. "
                "Features in red pushed this property toward secondary; blue toward primary."
            )

            # Waterfall chart
            shap_series = pd.Series(sv, index=feat_cols)
            top15 = shap_series.abs().sort_values(ascending=False).head(15).index
            wf = shap_series[top15].sort_values()

            colors = ["#F44336" if v > 0 else "#2196F3" for v in wf.values]
            fig = go.Figure(go.Bar(
                x=wf.values,
                y=wf.index,
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                title=f"SHAP Waterfall — Property {selected_idx}",
                xaxis_title="SHAP value (impact on prediction)",
                yaxis_title="Feature",
                height=450,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig, width="stretch")

            # Feature values table
            feat_table = pd.DataFrame({
                "Feature":       top15,
                "Value":         [f"{sample[f]:.3f}" for f in top15],
                "SHAP Impact":   [f"{sv[feat_cols.index(f)]:+.4f}" for f in top15],
            })
            st.dataframe(feat_table, width="stretch")
