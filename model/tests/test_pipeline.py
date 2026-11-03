"""Smoke tests for the training pipeline in `train_model_v1.py`.

These use a small slice of the real training data so they run in seconds,
without GCS or MLflow side effects.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

import ames_fastapi as api
from train_model_v1 import (
    CATEGORICAL_NONE_FILL,
    NUMERIC_NONE_FILL,
    build_feature_lists,
    build_hgb_pipeline,
    build_model_meta,
    build_rf_pipeline,
    evaluate,
    prepare_features,
)

TRAIN_DATA = 'input/train.csv'
pytestmark = pytest.mark.skipif(not os.path.exists(TRAIN_DATA), reason='training data missing')


@pytest.fixture()
def raw_df():
    return pd.read_csv(TRAIN_DATA)


@pytest.fixture()
def small_data(raw_df):
    X, y = prepare_features(raw_df)
    return X.head(250), y.head(250)


def test_prepare_features_drops_documented_outliers(raw_df):
    X, y = prepare_features(raw_df)
    assert len(X) == len(raw_df) - 4  # the 4 documented GrLivArea > 4000 rows
    assert (y > 0).all()


def test_prepare_features_domain_fill(raw_df):
    X, _ = prepare_features(raw_df)
    for col in NUMERIC_NONE_FILL:
        assert X[col].notna().all()
    for col in CATEGORICAL_NONE_FILL:
        assert X[col].notna().all()
        assert 'None' in X[col].unique()


def test_feature_lists(small_data):
    numeric, categorical = build_feature_lists(small_data[0])
    assert len(numeric) == 36
    assert len(categorical) == 43
    assert set(numeric) | set(categorical) == set(small_data[0].columns)


def test_rf_pipeline_predicts_dollars(small_data):
    X, y = small_data
    numeric, categorical = build_feature_lists(X)
    pipeline = build_rf_pipeline(
        numeric,
        categorical,
        n_estimators=30,
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
    assert (predictions > 1000).all()  # dollars after inverse log10 transform
    assert (np.log10(predictions) < 8).all()


def test_hgb_pipeline_predicts_dollars(small_data):
    X, y = small_data
    numeric, categorical = build_feature_lists(X)
    pipeline = build_hgb_pipeline(numeric, categorical, n_estimators=100)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert np.isfinite(predictions).all()
    assert (predictions > 1000).all()


def test_evaluate_returns_all_metrics(small_data):
    X, y = small_data
    numeric, categorical = build_feature_lists(X)
    pipeline = build_rf_pipeline(numeric, categorical, n_estimators=10)
    metrics = evaluate(pipeline, X, y, n_jobs=1)
    assert set(metrics) == {'r2', 'rmsle_dollars', 'rmse_log10', 'mape', 'fit_time_s'}
    assert metrics['rmsle_dollars'] > 0


def test_joblib_roundtrip(small_data, tmp_path):
    X, y = small_data
    numeric, categorical = build_feature_lists(X)
    pipeline = build_rf_pipeline(numeric, categorical, n_estimators=10)
    pipeline.fit(X, y)
    path = tmp_path / 'ames-rf-model.pkl'
    joblib.dump(pipeline, path, compress=3)
    loaded = joblib.load(path)
    np.testing.assert_array_max_ulp(pipeline.predict(X), loaded.predict(X), maxulp=2)


def test_model_meta_is_valid_json_contract(small_data):
    X, y = small_data
    meta = build_model_meta(X, 'rf', {'rmsle_dollars': 0.123}, run_id='test-run')
    assert meta['model_class'] == 'rf'
    assert meta['target_transform'] == 'identity'
    assert set(meta['features']) == set(X.columns)
    assert set(meta['feature_defaults']) == set(X.columns)
    json.dumps(meta)  # must be JSON-serializable for GCS upload


def test_price_from_raw_respects_serving_contract(small_data):
    X, _ = small_data
    meta = build_model_meta(X, 'rf', {}, run_id='test-run')
    api.model_meta = meta
    try:
        assert api.price_from_raw(150000.0) == pytest.approx(150000.0)
    finally:
        api.model_meta = None  # legacy mode
        assert api.price_from_raw(5.3) == pytest.approx(10**5.3)
