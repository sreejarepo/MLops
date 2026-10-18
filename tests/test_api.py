import joblib
import pytest
from fastapi import testclient

from src import api
from src.preprocessing import make_models

client = testclient.TestClient(api.app)


@pytest.fixture
def served_model(monkeypatch, tmp_path, sample_df):
    """Point the API at a freshly trained tiny model."""
    X, y = sample_df.drop(columns="target"), sample_df["target"]
    path = tmp_path / "model.joblib"
    joblib.dump(make_models(X)["logistic_regression"].fit(X, y), path)
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "MODEL_PATH", path)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(served_model, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease", "no_disease")
    assert 0.5 <= body["confidence"] <= 1.0
    probs = body["class_probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert body["confidence"] == pytest.approx(probs[body["label"]])
    assert "model_version" in body


def test_predict_rejects_missing_field(served_model, payload):
    del payload["age"]
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_unknown_field(served_model, payload):
    payload["favourite_colour"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_wrong_type(served_model, payload):
    payload["age"] = "not-a-number"
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_with_missing_model_returns_503(monkeypatch, tmp_path, payload):
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    assert client.post("/predict", json=payload).status_code == 503


def test_metrics_endpoint_exposes_prometheus_metrics(served_model, payload):
    client.get("/health")
    client.post("/predict", json=payload)
    text = client.get("/metrics/").text
    assert "api_requests_total" in text
    assert "api_request_duration_seconds" in text
    assert "predictions_total" in text
