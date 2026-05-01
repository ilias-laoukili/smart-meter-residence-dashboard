from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.back_end.config.settings import (
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_PARAMS,
    LOGS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.back_end.utils.feature_engineering import get_feature_columns


MODEL_REGISTRY: dict[str, tuple[Any, bool]] = {
    "Gradient Boosting": (
        GradientBoostingClassifier(**CLASSIFICATION_PARAMS),
        False,
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        False,
    ),
    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, C=1.0,
        ),
        True,
    ),
}


@dataclass
class ClassificationResult:
    """Container for all classification evaluation artifacts."""

    model_name: str
    model: Any
    accuracy: float
    f1: float
    roc_auc: float
    precision: float
    recall: float
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    feature_importance: pd.DataFrame
    report: str
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    cv_roc_auc_mean: float = field(default=0.0)
    cv_roc_auc_std: float = field(default=0.0)
    shap_values: np.ndarray | None = field(default=None)
    shap_expected: float | None = field(default=None)


def train_classifier(
    features_df: pd.DataFrame,
    model_name: str = "Gradient Boosting",
    params: dict | None = None,
) -> ClassificationResult:
    """Train and evaluate a classifier, return all artifacts."""
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0)
    y = features_df["residence_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    base_estimator, needs_scaling = MODEL_REGISTRY[model_name]

    if model_name == "Gradient Boosting" and params:
        base_estimator = GradientBoostingClassifier(
            **{**CLASSIFICATION_PARAMS, **params}
        )

    sample_weight = None
    if needs_scaling:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    base_estimator),
        ])
    else:
        # GBM doesn't expose class_weight so we pass sample_weight instead
        if model_name == "Gradient Boosting":
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            w_pos = (n_neg + n_pos) / (2 * n_pos) if n_pos > 0 else 1.0
            w_neg = (n_neg + n_pos) / (2 * n_neg) if n_neg > 0 else 1.0
            sample_weight = y_train.map({1: w_pos, 0: w_neg}).values
        clf = base_estimator

    if needs_scaling:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        inner_model = clf.named_steps["clf"]
    elif model_name == "Gradient Boosting":
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        inner_model = clf
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        inner_model = clf

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # 5-fold CV on training data only
    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
    )
    cv_roc_auc_mean = float(cv_scores.mean())
    cv_roc_auc_std = float(cv_scores.std())

    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
    elif hasattr(inner_model, "coef_"):
        importances = np.abs(inner_model.coef_[0])
    else:
        importances = np.zeros(len(feat_cols))

    importance_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    shap_values = None
    shap_expected = None
    if _SHAP_AVAILABLE and model_name in ("Gradient Boosting", "Random Forest"):
        try:
            explainer = _shap.TreeExplainer(inner_model)
            shap_out = explainer(X_test)
            if shap_out.values.ndim == 3:
                shap_values = shap_out.values[:, :, 1]   # class-1 slice
            else:
                shap_values = shap_out.values
            ev = explainer.expected_value
            if np.isscalar(ev):
                shap_expected = float(ev)
            elif len(ev) >= 2:
                shap_expected = float(ev[1])
            else:
                shap_expected = float(ev[0])
        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}", stacklevel=2)

    return ClassificationResult(
        model_name=model_name,
        model=clf,
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        confusion=confusion_matrix(y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        feature_importance=importance_df,
        report=classification_report(
            y_test, y_pred,
            target_names=["Principal", "Secondary"],
            zero_division=0,
        ),
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        cv_roc_auc_mean=cv_roc_auc_mean,
        cv_roc_auc_std=cv_roc_auc_std,
        shap_values=shap_values,
        shap_expected=shap_expected,
    )


def compare_models(features_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODEL_REGISTRY:
        r = train_classifier(features_df, model_name=name)
        rows.append({
            "Model": name,
            "Accuracy": round(r.accuracy, 4),
            "F1": round(r.f1, 4),
            "ROC-AUC": round(r.roc_auc, 4),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)


def save_model(result: ClassificationResult, path: str | None = None) -> Path:
    """Persist the trained model and log metrics."""
    model_path = Path(path) if path else CLASSIFICATION_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(result.model, f)

    log = {
        "model": result.model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "accuracy": result.accuracy,
        "f1": result.f1,
        "precision": result.precision,
        "recall": result.recall,
        "roc_auc": result.roc_auc,
        "cv_roc_auc_mean": result.cv_roc_auc_mean,
        "cv_roc_auc_std": result.cv_roc_auc_std,
        "confusion_matrix": result.confusion.tolist(),
        "top_features": result.feature_importance.head(10)["feature"].tolist(),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "classification_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return model_path


def load_model(path: str | None = None) -> Any:
    """Load a previously saved classifier."""
    model_path = Path(path) if path else CLASSIFICATION_MODEL_PATH
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_loaded_model(
    features_df: pd.DataFrame,
    path: str | None = None,
) -> ClassificationResult:
    """Load a saved classifier and evaluate it on the same held-out split."""
    model = load_model(path)
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0)
    y = features_df["residence_type"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feat_cols))

    importance_df = pd.DataFrame({
        "feature": feat_cols, "importance": importances,
    }).sort_values("importance", ascending=False)

    shap_values = shap_expected = None
    inner = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if _SHAP_AVAILABLE and hasattr(inner, "feature_importances_"):
        try:
            explainer = _shap.TreeExplainer(inner)
            shap_out = explainer(X_test)
            shap_values = (
                shap_out.values[:, :, 1]
                if shap_out.values.ndim == 3
                else shap_out.values
            )
            ev = explainer.expected_value
            if np.isscalar(ev):
                shap_expected = float(ev)
            elif len(ev) >= 2:
                shap_expected = float(ev[1])
            else:
                shap_expected = float(ev[0])
        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}", stacklevel=2)

    model_name = type(inner).__name__

    return ClassificationResult(
        model_name=model_name,
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        confusion=confusion_matrix(y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        feature_importance=importance_df,
        report=classification_report(
            y_test, y_pred,
            target_names=["Principal", "Secondary"],
            zero_division=0,
        ),
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        shap_values=shap_values,
        shap_expected=shap_expected,
    )
