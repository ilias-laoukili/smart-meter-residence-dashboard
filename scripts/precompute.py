"""
Pre-computation script — run once before deploying to Streamlit Cloud.

What it does
------------
1. Loads raw smart-meter data (~8.7 M rows) and the professor's label file.
2. Builds the per-property feature table (500 rows × ~30 features).
3. Saves the feature table as a Parquet file for fast dashboard loading.
4. Trains and saves the classification model (GBM, balanced).
5. Trains and saves the regression model (GBM occupancy rate).

Usage
-----
    python scripts/precompute.py

Run from the project root.  All artefacts go to:
    data/processed/features.parquet
    models/saved_models/classification_model.pkl
    models/saved_models/regression_model.pkl
    models/logs/classification_log.json
    models/logs/regression_log.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.back_end.config.settings import (
    CLASSIFICATION_MODEL_PATH,
    FEATURES_PARQUET_PATH,
    LABELS_CSV_PATH,
    RAW_CSV_PATH,
    REGRESSION_MODEL_PATH,
)
from src.back_end.training_pipelines.classification import (
    save_model as save_clf,
    train_classifier,
)
from src.back_end.training_pipelines.regression import (
    save_model as save_reg,
    train_regressor,
)
from src.back_end.utils.data_loader import load_labels, load_raw_data
from src.back_end.utils.feature_engineering import build_features


def _banner(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


def main() -> None:
    # ── Step 1 : Load raw data ─────────────────────────────────────────
    _banner("Step 1 / 5  —  Loading raw data")
    t0  = time.time()
    raw = load_raw_data(str(RAW_CSV_PATH))
    print(f"  Rows loaded : {len(raw):,}  ({time.time() - t0:.1f}s)")

    # ── Step 2 : Load professor's labels ──────────────────────────────
    _banner("Step 2 / 5  —  Loading ground-truth labels")
    labels = load_labels(str(LABELS_CSV_PATH))
    n_sec  = (labels["label"] == 1).sum()
    n_pri  = (labels["label"] == 0).sum()
    print(f"  Properties  : {len(labels)}  "
          f"(principal={n_pri}, secondary={n_sec})")

    # ── Step 3 : Feature engineering ──────────────────────────────────
    _banner("Step 3 / 5  —  Engineering features")
    t0       = time.time()
    features = build_features(raw, labels_df=labels)
    print(f"  Feature table: {features.shape}  ({time.time() - t0:.1f}s)")

    FEATURES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PARQUET_PATH, index=False)
    print(f"  Saved → {FEATURES_PARQUET_PATH}")

    # ── Step 4 : Train classifier ──────────────────────────────────────
    _banner("Step 4 / 5  —  Training Gradient Boosting classifier")
    t0     = time.time()
    clf_r  = train_classifier(features, model_name="Gradient Boosting")
    save_clf(clf_r)
    print(
        f"  Accuracy={clf_r.accuracy:.3f}  "
        f"F1={clf_r.f1:.3f}  "
        f"ROC-AUC={clf_r.roc_auc:.3f}  "
        f"({time.time() - t0:.1f}s)"
    )
    print(f"  Saved → {CLASSIFICATION_MODEL_PATH}")

    # ── Step 5 : Train regressor ───────────────────────────────────────
    _banner("Step 5 / 5  —  Training Gradient Boosting regressor")
    t0    = time.time()
    reg_r = train_regressor(features)
    save_reg(reg_r)
    print(
        f"  MAE={reg_r.mae:.4f}  "
        f"RMSE={reg_r.rmse:.4f}  "
        f"R²={reg_r.r2:.4f}  "
        f"({time.time() - t0:.1f}s)"
    )
    print(f"  Saved → {REGRESSION_MODEL_PATH}")

    _banner("Done!  All artefacts saved.")
    print(
        "\n  You can now commit data/processed/ and models/ to the repo,\n"
        "  or rely on Streamlit Cloud's @st.cache_resource to train on\n"
        "  first launch.\n"
    )


if __name__ == "__main__":
    main()
