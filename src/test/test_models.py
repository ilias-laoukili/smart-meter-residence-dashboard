"""Tests for classification and regression training pipelines."""
import pickle
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from src.back_end.training_pipelines.classification import (
    ClassificationResult,
    train_classifier,
    save_model,
    load_model,
)
from src.back_end.training_pipelines.regression import (
    RegressionResult,
    apply_inhabited_threshold,
    train_regressor,
    save_model as reg_save_model,
    load_model as reg_load_model,
    predict_full_dataset,
)
from src.back_end.utils.feature_engineering import build_features, get_feature_columns

def _make_raw_for_features(n_days=60, n_meters=40):
    import pandas as _pd
    records = []
    rng = np.random.default_rng(0)
    for i in range(n_meters):
        meter = f"M{i:03d}"
        is_secondary = i >= (n_meters * 3 // 4)
        for day_offset in range(n_days):
            day_dt = _pd.Timestamp("2024-01-01", tz="UTC") + _pd.Timedelta(days=day_offset)
            for slot in range(48):
                ts = day_dt + _pd.Timedelta(minutes=30 * slot)
                if is_secondary and day_offset % 7 < 5:
                    val = 0.0  # mostly absent
                else:
                    val = float(rng.uniform(50, 500))
                records.append({"id": meter, "horodate": ts, "valeur": val})
    return _pd.DataFrame(records)


def _make_labels(n_meters=40):
    rows = []
    for i in range(n_meters):
        rows.append({
            "id": f"M{i:03d}",
            "label": 1 if i >= (n_meters * 3 // 4) else 0,
            "cluster": i % 5,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def features_df():
    raw = _make_raw_for_features()
    labels = _make_labels()
    return build_features(raw, labels_df=labels)

def test_train_classifier_returns_result(features_df):
    result = train_classifier(features_df, model_name="Gradient Boosting")
    assert isinstance(result, ClassificationResult)


def test_classification_metrics_bounded(features_df):
    result = train_classifier(features_df, model_name="Gradient Boosting")
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0


def test_classification_cv_fields(features_df):
    result = train_classifier(features_df, model_name="Gradient Boosting")
    assert 0.0 <= result.cv_roc_auc_mean <= 1.0
    assert result.cv_roc_auc_std >= 0.0


def test_classification_confusion_matrix_shape(features_df):
    result = train_classifier(features_df, model_name="Gradient Boosting")
    assert result.confusion.shape == (2, 2)


def test_classification_feature_importance_nonempty(features_df):
    result = train_classifier(features_df, model_name="Gradient Boosting")
    assert len(result.feature_importance) > 0
    assert "feature" in result.feature_importance.columns
    assert "importance" in result.feature_importance.columns


def test_classification_save_load_roundtrip(features_df, tmp_path):
    result = train_classifier(features_df, model_name="Gradient Boosting")
    path = str(tmp_path / "clf.pkl")
    save_model(result, path=path)
    loaded = load_model(path=path)
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0)
    preds = loaded.predict(X)
    assert len(preds) == len(features_df)


def test_classification_log_written(features_df, tmp_path, monkeypatch):
    import src.back_end.training_pipelines.classification as clf_module
    monkeypatch.setattr(clf_module, "LOGS_DIR", tmp_path)
    result = train_classifier(features_df, model_name="Gradient Boosting")
    save_model(result, path=str(tmp_path / "clf.pkl"))
    import json
    log = json.loads((tmp_path / "classification_log.json").read_text())
    for key in ("model", "trained_at", "accuracy", "f1", "roc_auc",
                "cv_roc_auc_mean", "confusion_matrix", "top_features"):
        assert key in log, f"missing key: {key}"


def test_random_forest_classifier(features_df):
    result = train_classifier(features_df, model_name="Random Forest")
    assert isinstance(result, ClassificationResult)
    assert 0.0 <= result.roc_auc <= 1.0


def test_logistic_regression_classifier(features_df):
    result = train_classifier(features_df, model_name="Logistic Regression")
    assert isinstance(result, ClassificationResult)
    assert 0.0 <= result.accuracy <= 1.0

def test_train_regressor_returns_result(features_df):
    result = train_regressor(features_df)
    assert isinstance(result, RegressionResult)


def test_regression_metrics_reasonable(features_df):
    result = train_regressor(features_df)
    assert result.mae >= 0.0
    assert result.rmse >= 0.0


def test_regression_r2_bounded(features_df):
    result = train_regressor(features_df)
    assert result.r2 <= 1.0


def test_regression_residuals_shape(features_df):
    result = train_regressor(features_df)
    assert len(result.residuals) == len(result.y_test)


def test_regression_save_load_roundtrip(features_df, tmp_path):
    result = train_regressor(features_df)
    path = str(tmp_path / "reg.pkl")
    reg_save_model(result, path=path)
    loaded = reg_load_model(path=path)
    feat_cols = get_feature_columns(features_df)
    X = features_df[feat_cols].fillna(0)
    preds = loaded.predict(X)
    assert len(preds) == len(features_df)


def test_regression_log_written(features_df, tmp_path, monkeypatch):
    import src.back_end.training_pipelines.regression as reg_module
    monkeypatch.setattr(reg_module, "LOGS_DIR", tmp_path)
    result = train_regressor(features_df)
    reg_save_model(result, path=str(tmp_path / "reg.pkl"))
    import json
    log = json.loads((tmp_path / "regression_log.json").read_text())
    for key in ("trained_at", "mae", "rmse", "r2", "top_features"):
        assert key in log, f"missing key: {key}"


def test_predict_full_dataset_clipped(features_df):
    result = train_regressor(features_df)
    out = predict_full_dataset(result, features_df)
    assert out["occupancy_rate_pred"].between(0, 1).all()

def test_apply_inhabited_threshold_basic():
    scores = np.array([0.0, 0.5, 0.9, 1.0])
    result = apply_inhabited_threshold(scores, threshold=0.8)
    np.testing.assert_array_equal(result, [0, 0, 1, 1])


def test_apply_inhabited_threshold_boundary():
    scores = np.array([0.95])
    assert apply_inhabited_threshold(scores, threshold=0.95)[0] == 1
    assert apply_inhabited_threshold(scores, threshold=0.96)[0] == 0


def test_apply_inhabited_threshold_all_zeros():
    scores = np.zeros(10)
    assert apply_inhabited_threshold(scores, threshold=0.5).sum() == 0
