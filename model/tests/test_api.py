import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from ames_fastapi import app


@pytest.fixture(autouse=True)
def setup_test_model():
    """Inject a minimal RandomForest model so the app can start and predict."""

    dummy = RandomForestRegressor(n_estimators=2, random_state=42)
    dummy.fit(np.random.rand(10, 10), np.random.rand(10))

    import ames_fastapi as m

    m.model = dummy
    m.feature_names = [f'f{i}' for i in range(10)]
    yield


client = TestClient(app)


def test_health():
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert data['features_count'] == 10


def test_predict():
    payload = {
        'OverallQual': 8,
        'GrLivArea': 2000,
        'GarageCars': 2.0,
        'GarageArea': 600.0,
        'TotalBsmtSF': 1200.0,
        'FirstFlrSF': 1200,
        'YearBuilt': 2000,
        'YearRemodAdd': 2000,
        'FullBath': 3,
        'TotRmsAbvGrd': 8,
    }
    resp = client.post('/predict', json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 'predicted_price' in data
    assert isinstance(data['predicted_price'], float)
    assert data['predicted_price'] > 0


def test_predict_invalid_type():
    resp = client.post('/predict', json={'OverallQual': 'bad'})
    assert resp.status_code == 422


def test_root():
    resp = client.get('/')
    assert resp.status_code == 200
    data = resp.json()
    assert 'endpoints' in data
