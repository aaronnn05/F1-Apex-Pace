import pytest
from fastapi.testclient import TestClient

from apex_pace.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200

def test_prediction(client):
    response = client.post("/predict", json={
        "prev_lap_time": 98.5,
        "rolling_3lap_mean": 98.6,
        "compound_code": 3,
        "is_fresh_tyre": 0,
        "TyreLife": 15,
    })

    assert response.status_code == 200
    data = response.json()
    assert "predicted_lap_time_seconds" in data

def test_invalid_tyre_life(client):
    response = client.post("/predict", json={
        "prev_lap_time": 98.5,
        "rolling_3lap_mean": 98.6,
        "compound_code": 3,
        "is_fresh_tyre": 0,
        "TyreLife": -1,
    })

    assert response.status_code == 422

def test_invalid_compound(client):
    response = client.post("/predict", json={
        "prev_lap_time": 98.5,
        "rolling_3lap_mean": 98.6,
        "compound_code": 10,
        "is_fresh_tyre": 0,
        "TyreLife": 15,
    })

    assert response.status_code == 422