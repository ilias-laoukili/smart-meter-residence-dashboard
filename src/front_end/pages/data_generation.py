from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA as _PCA
from sklearn.manifold import TSNE as _TSNE
from sklearn.preprocessing import StandardScaler
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.back_end.config.settings import FEATURES_PARQUET_PATH
from src.back_end.training_pipelines.generative import (
    compare_distributions,
    generate_autoencoder,
    generate_diffusion,
    generate_gan,
    synthetic_to_csv_bytes,
    train_autoencoder,
    train_diffusion,
    train_gan,
    wasserstein_distances,
)
from src.back_end.utils.data_loader import load_labels, load_raw_data
from src.back_end.utils.feature_engineering import build_features, get_feature_columns
# Cached data

@st.cache_data(show_spinner="Loading features…")
def _load_features() -> pd.DataFrame:
    if FEATURES_PARQUET_PATH.exists():
        return pd.read_parquet(FEATURES_PARQUET_PATH)
    raw    = load_raw_data()
    labels = load_labels()
    return build_features(raw, labels_df=labels)
# Page

st.title("Synthetic Data Generation")
st.markdown(
    "Generate synthetic property features using three generative architectures "
    "and evaluate how faithfully they reproduce the real distribution.  \n\n"
    "**Quality metric:** 1-D Wasserstein distance per feature "
    "(lower = more faithful).  Samples can be downloaded as CSV."
)
st.info(
    "With only 500 properties and just 72 secondary residences (14 %), training data is scarce. "
    "Synthetic generation creates new plausible property profiles that match the statistical "
    "patterns of real ones — useful for augmenting the minority class or sharing data without "
    "exposing real meter IDs."
)
st.markdown(
    "**When to use each model:** Autoencoder is fastest and most stable. "
    "GAN produces more varied samples but can be unstable. "
    "Diffusion model is slowest but highest quality."
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
feat_cols = get_feature_columns(features)
X_real    = features[feat_cols].fillna(0).values
st.sidebar.header("Generation Settings")
n_samples  = st.sidebar.slider("Synthetic samples", 50, 500, 200, step=50)
projection = st.sidebar.radio("Projection method", ["pca", "tsne"], horizontal=True)

# Session state for generated data
for key in ("gen_ae", "gen_gan", "gen_diff"):
    if key not in st.session_state:
        st.session_state[key] = None


def _quality_section(
    synthetic: np.ndarray,
    model_label: str,
) -> None:
    """Shared quality display: projection scatter + Wasserstein table + download."""
    df_proj = compare_distributions(X_real, synthetic, method=projection)
    fig = px.scatter(
        df_proj, x="x", y="y", color="source",
        title=f"Real vs. {model_label} ({projection.upper()})",
        opacity=0.6,
        color_discrete_map={"Real": "#2196F3", "Synthetic": "#FF5722"},
    )
    st.plotly_chart(fig, width="stretch")

    w_df = wasserstein_distances(X_real, synthetic, feat_cols)
    mean_w = w_df["wasserstein_distance"].mean()
    st.metric("Mean Wasserstein Distance (normalised)", f"{mean_w:.4f}",
              help="Computed on z-scored features — dimensionless and scale-invariant. Lower = better. A value < 0.5 is good.")

    with st.expander("Per-feature Wasserstein distances"):
        fig2 = px.bar(
            w_df.sort_values("wasserstein_distance"),
            x="wasserstein_distance", y="feature",
            orientation="h",
            title=f"Wasserstein Distance per Feature — {model_label}",
            color="wasserstein_distance",
            color_continuous_scale="Reds",
        )
        fig2.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, width="stretch")
        st.dataframe(w_df, width="stretch")

    csv_bytes = synthetic_to_csv_bytes(synthetic, feat_cols)
    st.download_button(
        label=f"Download {model_label} synthetic data (CSV)",
        data=csv_bytes,
        file_name=f"synthetic_{model_label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
tab_ae, tab_gan, tab_diff, tab_compare = st.tabs([
    "Autoencoder",
    "GAN",
    "Diffusion Model",
    "Compare All",
])
# autoencoder
with tab_ae:
    st.subheader("Autoencoder")
    st.markdown(
        "Learns a compressed **latent representation** of real features and "
        "decodes new samples from randomly sampled latent vectors.  "
        "Good at capturing the overall data manifold."
    )

    col1, col2 = st.columns(2)
    latent_dim = col1.number_input("Latent dim", 4, 32, 8, key="ae_ld")
    epochs_ae  = col2.number_input("Epochs", 50, 1000, 200, step=50, key="ae_ep")

    if st.button("Train & Generate", key="btn_ae", type="primary"):
        with st.spinner("Training autoencoder…"):
            model_ae, scaler_ae, losses_ae = train_autoencoder(
                X_real, latent_dim=int(latent_dim), epochs=int(epochs_ae)
            )
            synthetic_ae = generate_autoencoder(
                model_ae, scaler_ae, n_samples, int(latent_dim)
            )
            st.session_state["gen_ae"] = synthetic_ae

        fig = px.line(
            y=losses_ae, title="Autoencoder Training Loss",
            labels={"x": "Epoch", "y": "MSE Loss"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "A smooth, steadily falling curve means the model is learning. "
            "If the loss stops improving early, try more epochs or a larger latent dimension."
        )

    if st.session_state["gen_ae"] is not None:
        st.divider()
        _quality_section(st.session_state["gen_ae"], "Autoencoder")
# gan
with tab_gan:
    st.subheader("GAN (Generative Adversarial Network)")
    st.markdown(
        "A **generator** learns to fool a **discriminator**, producing "
        "increasingly realistic feature vectors.  Works well for mode coverage "
        "but can suffer from training instability (watch the loss curves)."
    )

    col1, col2 = st.columns(2)
    noise_dim  = col1.number_input("Noise dim", 8, 64, 16, key="gan_nd")
    epochs_gan = col2.number_input("Epochs", 100, 2000, 300, step=100, key="gan_ep")

    if st.button("Train & Generate", key="btn_gan", type="primary"):
        with st.spinner("Training GAN…"):
            gen_model, scaler_g, g_losses, d_losses = train_gan(
                X_real, noise_dim=int(noise_dim), epochs=int(epochs_gan)
            )
            synthetic_gan = generate_gan(
                gen_model, scaler_g, n_samples, int(noise_dim)
            )
            st.session_state["gen_gan"] = synthetic_gan

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=g_losses, name="Generator",     line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(y=d_losses, name="Discriminator", line=dict(color="#FF5722")))
        fig.update_layout(
            title="GAN Training Losses",
            xaxis_title="Epoch",
            yaxis_title="Loss",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Watch both lines: the generator loss should fall while the discriminator stays moderate. "
            "If the discriminator loss drops to near zero and stays there, the generator has failed — "
            "try fewer epochs or a smaller noise dimension."
        )

    if st.session_state["gen_gan"] is not None:
        st.divider()
        _quality_section(st.session_state["gen_gan"], "GAN")
# diffusion
with tab_diff:
    st.subheader("Diffusion Model (Experimental)")
    st.markdown(
        "A simplified **DDPM-style** model for tabular data.  During training "
        "the model learns to predict noise added at each timestep; generation "
        "iteratively denoises pure Gaussian noise into realistic samples.  "
        "Slower than AE / GAN but state-of-the-art in generation quality."
    )

    col1, col2 = st.columns(2)
    timesteps  = col1.number_input("Timesteps", 50, 500, 100, step=50, key="diff_ts")
    epochs_diff = col2.number_input("Epochs", 50, 1000, 200, step=50, key="diff_ep")

    if st.button("Train & Generate", key="btn_diff", type="primary"):
        with st.spinner("Training diffusion model…"):
            denoiser, scaler_d, betas, losses_d = train_diffusion(
                X_real, timesteps=int(timesteps), epochs=int(epochs_diff)
            )
            synthetic_diff = generate_diffusion(
                denoiser, scaler_d, betas, n_samples, X_real.shape[1]
            )
            st.session_state["gen_diff"] = synthetic_diff

        fig = px.line(
            y=losses_d, title="Diffusion Training Loss",
            labels={"x": "Epoch", "y": "MSE Loss"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Diffusion models train more slowly than the others — 500+ epochs recommended. "
            "The loss should decrease smoothly without spikes."
        )

    if st.session_state["gen_diff"] is not None:
        st.divider()
        _quality_section(st.session_state["gen_diff"], "Diffusion")
# compare all
with tab_compare:
    st.subheader("Side-by-Side Comparison")

    generated: dict[str, np.ndarray] = {}
    if st.session_state["gen_ae"]   is not None: generated["Autoencoder"] = st.session_state["gen_ae"]
    if st.session_state["gen_gan"]  is not None: generated["GAN"]         = st.session_state["gen_gan"]
    if st.session_state["gen_diff"] is not None: generated["Diffusion"]   = st.session_state["gen_diff"]

    if not generated:
        st.info("Train at least one generative model above to see comparisons.")
    else:
        # Wasserstein summary table
        st.caption(
            "Wasserstein distance measures how different the synthetic data is from the real data "
            "for each feature. Lower = more faithful. A mean below 0.3 is good; above 0.6 suggests "
            "the model has not learned that feature well."
        )
        st.markdown("#### Wasserstein Distance Summary (mean across all features)")
        w_rows = []
        for name, synth in generated.items():
            w_df  = wasserstein_distances(X_real, synth, feat_cols)
            w_rows.append({
                "Model":                    name,
                "Mean Wasserstein":         round(w_df["wasserstein_distance"].mean(), 4),
                "Median Wasserstein":       round(w_df["wasserstein_distance"].median(), 4),
                "Max Wasserstein (worst feature)": round(w_df["wasserstein_distance"].max(), 4),
            })
        st.dataframe(pd.DataFrame(w_rows).set_index("Model"), width="stretch")

        # Feature-wise mean comparison
        st.markdown("#### Feature-wise Mean Comparison")
        comp_rows = [{"Source": "Real", **dict(zip(feat_cols, X_real.mean(axis=0)))}]
        for name, synth in generated.items():
            comp_rows.append({"Source": name, **dict(zip(feat_cols, synth.mean(axis=0)))})
        comp_df = pd.DataFrame(comp_rows).set_index("Source")
        st.dataframe(comp_df.T.round(3), width="stretch")

        # Joint projection
        st.markdown("#### Joint Projection")
        all_synth  = np.vstack(list(generated.values()))
        all_labels = []
        for name, synth in generated.items():
            all_labels.extend([name] * len(synth))

        combined        = np.vstack([X_real, all_synth])
        labels_all      = ["Real"] * len(X_real) + all_labels
        combined_scaled = StandardScaler().fit_transform(combined)

        if projection == "pca":
            proj = _PCA(n_components=2, random_state=42).fit_transform(combined_scaled)
        else:
            perp = min(30, len(combined_scaled) - 1)
            proj = _TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(
                combined_scaled
            )

        proj_df = pd.DataFrame({
            "x": proj[:, 0], "y": proj[:, 1], "Source": labels_all
        })
        fig = px.scatter(
            proj_df, x="x", y="y", color="Source",
            title=f"All Models — Real vs. Synthetic ({projection.upper()})",
            opacity=0.55,
        )
        st.plotly_chart(fig, width="stretch")
