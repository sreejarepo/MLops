from fastapi.testclient import TestClient

from src import api

client = TestClient(api.app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_with_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    payload = {"age": 58, "sex": 1, "cp": 3, "trestbps": 140, "chol": 240,
               "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
               "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
