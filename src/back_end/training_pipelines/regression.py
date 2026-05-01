from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.back_end.config.settings import (
    LOGS_DIR,
    RANDOM_STATE,
    REGRESSION_MODEL_PATH,
    REGRESSION_PARAMS,
    TEST_SIZE,
)
from src.back_end.utils.feature_engineering import get_feature_columns


@dataclass
class RegressionResult:
    """Container for regression evaluation artifacts."""

    model: Any
    mae: float
    rmse: float
    r2: float
    feature_importance: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    residuals: np.ndarray


def train_regressor(
    features_df: pd.DataFrame,
    params: dict | None = None,
) -> RegressionResult:
    """Train a GradientBoosting regressor on occupancy_rate."""
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0)
    y = features_df["occupancy_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    hp = {**REGRESSION_PARAMS, **(params or {})}
    model = GradientBoostingRegressor(**hp)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return RegressionResult(
        model=model,
        mae=mean_absolute_error(y_test, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        r2=r2_score(y_test, y_pred),
        feature_importance=importance,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        residuals=residuals,
    )


def predict_full_dataset(
    result: RegressionResult,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0)
    preds = result.model.predict(X)

    out = features_df[["id"]].copy()
    out["occupancy_rate_pred"] = preds.clip(0, 1)
    if "residence_type" in features_df.columns:
        out["residence_type"] = features_df["residence_type"].values
    if "cluster" in features_df.columns:
        out["cluster"] = features_df["cluster"].values
    return out


def apply_inhabited_threshold(
    occupancy_scores: pd.Series | np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    arr = np.asarray(occupancy_scores)
    return (arr >= threshold).astype(int)


def save_model(result: RegressionResult, path: str | None = None) -> Path:
    """Persist the trained model and log metrics."""
    model_path = Path(path) if path else REGRESSION_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(result.model, f)

    log = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "mae": result.mae,
        "rmse": result.rmse,
        "r2": result.r2,
        "top_features": result.feature_importance.head(10)["feature"].tolist(),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "regression_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return model_path


def load_model(path: str | None = None) -> Any:
    """Load a previously saved regressor."""
    model_path = Path(path) if path else REGRESSION_MODEL_PATH
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_loaded_model(
    features_df: pd.DataFrame,
    path: str | None = None,
) -> RegressionResult:
    """Load a saved regressor and evaluate it on the same held-out split."""
    model = load_model(path)
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0)
    y = features_df["occupancy_rate"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return RegressionResult(
        model=model,
        mae=mean_absolute_error(y_test, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        r2=r2_score(y_test, y_pred),
        feature_importance=importance,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        residuals=residuals,
    )
