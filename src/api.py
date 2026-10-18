"""FastAPI prediction API with request logging and Prometheus metrics."""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart-disease-api")
REQUESTS = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_duration_seconds", "Request duration", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions served, by predicted class", ["label"])

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/heart_disease_model.joblib"))
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
app.mount("/metrics", make_asgi_app())
_model = None


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        endpoint = request.url.path
        REQUESTS.labels(request.method, endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(duration)
        logger.info(json.dumps({"event": "request", "method": request.method,
                                "path": endpoint, "status": status,
                                "duration_seconds": round(duration, 6)}))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientFeatures) -> dict[str, Any]:
    try:
        model = get_model()
        frame = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.classes_)
        probs = {("disease" if int(cls) == 1 else "no_disease"): float(prob)
                 for cls, prob in zip(classes, probabilities)}
        label = "disease" if prediction == 1 else "no_disease"
        confidence = probs[label]  # probability of the predicted class
        PREDICTIONS.labels(label).inc()
        logger.info(json.dumps({"event": "prediction", "prediction": prediction,
                                "label": label, "confidence": round(confidence, 4)}))
        return {"prediction": prediction, "label": label, "confidence": confidence,
                "class_probabilities": probs,
                "model_version": os.getenv("MODEL_VERSION", "1.0.0"),
                "notice": "Educational model only; not a medical diagnosis."}
    except FileNotFoundError as exc:
        logger.exception("Model artifact missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
