# Smart Meter Residence Classification

Streamlit dashboard analysing electricity smart-meter data (500 properties, ~8.7 M readings, Nov 2023 – Oct 2024).

## Objectives

Two prediction tasks from the same consumption data:

1. **Classify residence type** — is a property a *primary residence* (permanently occupied, regular year-round usage) or a *secondary residence* (holiday home, occasional use, long absence periods)?  
   Ground-truth labels come from the professor's annotation file (`RES2-6-9-labels.csv`).

2. **Determine if the dwelling is currently inhabited** — based on consumption-derived metrics (fraction of active days, consecutive zero-consumption days, seasonal regularity), a regression model produces an occupancy score ∈ [0, 1]. A user-adjustable threshold converts this score to a binary **inhabited / not-inhabited** flag.

The dashboard also includes **SHAP-based interpretations** that explain, for any property, which consumption patterns drove the model's prediction.

---

## Dataset

| | |
|---|---|
| Source | `data/raw/export.csv` |
| Readings | ~8.7 M half-hourly readings |
| Properties | 500 unique meters |
| Period | November 2023 – October 2024 |
| Columns | `id` (meter ID), `horodate` (UTC timestamp), `valeur` (Wh) |
| Labels | `data/raw/RES2-6-9-labels.csv` — 428 principal, 72 secondary, clusters 0–9 |

---

## Setup

```bash
git clone https://github.com/ilias-laoukili/smart-meter-residence-dashboard.git
cd smart-meter-residence-dashboard
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pre-compute features and pre-train models once (takes ~5–10 min on first run):

```bash
python scripts/precompute.py
streamlit run src/front_end/app.py
```

Without `precompute.py`, the dashboard computes everything on first load using Streamlit's cache — slower but functional.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                        # Raw CSV files (not tracked by git)
│   └── processed/
│       └── features.parquet        # Pre-computed feature table (500 × ~30)
├── models/
│   ├── saved_models/               # Trained classifier + regressor (.pkl)
│   └── logs/                       # Training metrics (JSON)
├── notebooks/
│   └── 01_clustering/              # Exploratory clustering notebook
├── scripts/
│   └── precompute.py               # Pre-computation script
├── src/
│   ├── back_end/
│   │   ├── config/settings.py      # Paths, constants, hyperparameters
│   │   ├── utils/
│   │   │   ├── data_loader.py
│   │   │   └── feature_engineering.py
│   │   └── training_pipelines/
│   │       ├── classification.py   # GBM / RF / LR + SHAP
│   │       ├── regression.py       # GBM regressor + occupancy threshold
│   │       └── generative.py       # Autoencoder, GAN, Diffusion model
│   ├── front_end/
│   │   ├── app.py                  # Streamlit entry point
│   │   └── pages/
│   │       ├── home.py
│   │       ├── classification.py
│   │       ├── regression.py
│   │       ├── data_generation.py
│   │       └── explorer.py
│   └── test/
│       ├── test_data.py
│       ├── test_features.py
│       └── test_models.py
├── requirements.txt
└── README.md
```

---

## Feature Engineering

Raw half-hourly readings are aggregated to one row per property (~30 features):

| Category | Features |
|---|---|
| Global stats | `mean_consumption`, `std_consumption`, `cv_consumption`, `total_consumption` |
| Time-of-day | `mean_night`, `mean_morning`, `mean_afternoon`, `mean_evening`, `night_baseline` |
| Seasonal | `winter_mean`, `summer_mean`, `winter_summer_ratio` |
| Behavioural | `weekday_mean`, `weekend_mean`, `weekend_weekday_ratio`, `peak_share` (17–21 h) |
| Zero-days | `zero_days_ratio`, `summer_zero_ratio`, `winter_zero_ratio`, `max_consecutive_zero_days` |
| Regularity | `daily_autocorr`, `active_weeks_ratio`, `monthly_entropy` |

The strongest signals for secondary residences: high `max_consecutive_zero_days` (long absences), low `monthly_entropy` (consumption concentrated in a few months), low `night_baseline` (no always-on appliances).

**Metrics used to determine occupancy:** the regression target `occupancy_rate` is the fraction of days with total consumption > 500 Wh — a threshold above standby loads but well below any occupied household. Supporting features: `zero_days_ratio`, `max_consecutive_zero_days`, `active_weeks_ratio`, and `daily_autocorr`. The target is computed directly from raw daily totals (not from aggregated features) to prevent target leakage.

---

## Models

### Classification — Principal vs. Secondary Residence

Three algorithms available for comparison:

| Model | Imbalance handling |
|---|---|
| **Gradient Boosting** (default) | Manual sample-weight balancing |
| **Random Forest** | `class_weight='balanced'` |
| **Logistic Regression** | `class_weight='balanced'` |

The dataset has 85.6 % / 14.4 % class split. Gradient Boosting was chosen over neural networks because tree-based models consistently outperform MLPs on small tabular datasets (400 training samples, 27 features). SHAP TreeExplainer provides exact global (beeswarm) and per-property (waterfall) explanations for the two tree models.

### Regression — Occupancy Rate

A Gradient Boosting regressor predicts a continuous score ∈ [0, 1]. A user-adjustable threshold (default 0.95) converts this to a binary **inhabited / not-inhabited** flag. The default is 0.95 rather than 0.5 because this dataset's occupancy distribution is highly skewed toward 1.0 — a threshold of 0.5 would classify every property as inhabited.

### Synthetic Data Generation

The dataset contains only 72 secondary residences (14.4 % of 500 properties). This class imbalance limits classifier training and makes it difficult to study secondary-residence patterns in isolation. Three generative models are trained on the feature table to produce new synthetic property profiles that match the statistical distribution of real ones:

| Model | Notes |
|---|---|
| **Autoencoder** | Encodes features to a latent space, decodes random latent vectors. Stable, good global fit. |
| **GAN** | Generator vs. discriminator adversarial training. More varied samples, can be unstable. |
| **Diffusion (DDPM)** | Iterative denoising from Gaussian noise. Best quality, slowest to train. |

Quality is measured with the 1-D Wasserstein distance per feature, computed on z-scored data to make distances scale-invariant. Generated samples can be downloaded as CSV.

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Home & KPIs** | Dataset overview, label distribution, seasonal patterns, cluster analysis, correlation heatmap |
| **Classification** | Train or load a model, confusion matrix, ROC curve, SHAP global + per-property |
| **Regression** | Occupancy predictions, inhabited threshold slider, breakdown by residence type and cluster |
| **Data Generation** | Train AE / GAN / Diffusion, Wasserstein quality metrics, side-by-side comparison, CSV export |
| **Property Explorer** | Per-meter time series, hour×month heatmap, seasonal profile, feature radar |

---

## Results

Typical metrics after running `scripts/precompute.py` on the full dataset:

| Metric | Value |
|---|---|
| Classifier ROC-AUC | ~0.85–0.92 |
| Classifier F1 (secondary class) | ~0.60–0.75 |
| Regressor R² | ~0.85–0.95 |

---

## Tests

```bash
python -m pytest src/test/ -v
```

Tests cover data loading, feature engineering, and model train/save/load roundtrips. They use small synthetic DataFrames and do not require the raw CSV.
