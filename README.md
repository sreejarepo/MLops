# Heart Disease MLOps

End-to-end MLOps project on the UCI Heart Disease dataset: EDA, model training and
tuning, MLflow experiment tracking, tests + CI/CD, Docker, Kubernetes deployment and
Prometheus/Grafana monitoring. **Educational project - not a clinical diagnostic system.**

## Repository layout

```
download_data.py        fetch UCI dataset (id 45) -> data/raw/heart_disease.csv
src/
  data.py               load + validate + binarise target
  clean.py              cleaned dataset -> data/processed/ (+ data-quality report)
  preprocessing.py      leakage-safe ColumnTransformer + model pipelines
  evaluation.py         test metrics, confusion matrix / ROC / report artifacts
  tracking.py           single MLflow configuration (sqlite by default)
  train.py              baselines + 5-fold CV, logs everything to MLflow
  tune.py               GridSearchCV tuning, promotes model only if it improves CV AUC
  inference.py          batch inference CLI
  api.py                FastAPI service (/predict, /health, /metrics/)
notebooks/01_eda.ipynb  EDA (executed, outputs saved)
tests/                  pytest suite (data, cleaning, model, train, tune, inference, API)
Dockerfile              API image (runtime deps only: requirements.txt)
docker-compose.yml      API + Prometheus + Grafana
k8s/deployment.yaml     Deployment (2 replicas, probes) + LoadBalancer Service
monitoring/             prometheus.yml + provisioned Grafana datasource and dashboard
.github/workflows/      ci.yml (lint, test, train, docker) and tune.yml (manual)
report/, screenshots/   written report and evidence
```

## 1. Setup (clean machine)

Python **3.11** (same version as Docker and CI).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Data

```bash
python download_data.py      # UCI id 45 via ucimlrepo; retries; keeps the committed CSV if UCI is unreachable
python -m src.clean          # -> data/processed/heart_disease_clean.csv + data_quality_report.json
```

303 rows x 13 features, binary target (0 = no disease, 1-4 -> 1). Only `ca` (4) and `thal` (2)
have missing values. Training starts from the **raw** CSV and imputes inside the sklearn
pipeline (fit on training folds only) to avoid leakage; `clean.py` produces an inspectable
fully-cleaned copy.

## 3. EDA

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## 4. Train, tune, track

```bash
python -m src.train     # LR + RF baselines, 5-fold CV, test metrics, plots, model logged to MLflow
python -m src.tune      # GridSearchCV for both models; promotes a tuned model only if it beats the baseline
```

Both scripts log to the same MLflow backend (`sqlite:///mlflow.db`, override with the
`MLFLOW_TRACKING_URI` environment variable):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 - experiments `heart-disease-classification` and `heart-disease-tuning`.
Every run logs parameters, CV + test metrics, confusion matrix / ROC / classification report,
(train) EDA plots and `requirements.txt`, (tune) the full grid as CSV, and the fitted pipeline.
`reports/final_model.json` records which model is deployed and where it came from.

## 5. Tests and lint

```bash
ruff check src tests download_data.py
pytest -v
```

## 6. Run the API locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Docs http://localhost:8000/docs - health `/health` - metrics `/metrics/`

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":3,"trestbps":140,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

```json
{"prediction": 0, "label": "no_disease", "confidence": 0.52,
 "class_probabilities": {"no_disease": 0.52, "disease": 0.48},
 "model_version": "1.0.0", "notice": "Educational model only; not a medical diagnosis."}
```

`confidence` is the model's probability for the predicted class (not a calibrated clinical risk).

## 7. Batch inference

```bash
python -m src.inference --sample
python -m src.inference --input my_patients.csv --output predictions.csv
```

## 8. Docker

```bash
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0
```

API + Prometheus + Grafana:

```bash
docker compose up --build
```

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8002 | `/docs`, `/predict`, `/metrics/` |
| Prometheus | http://localhost:9090 | Status -> Targets must show `heart-disease-api` UP |
| Grafana | http://localhost:3000 | login `admin` / `admin`; dashboard "Heart Disease API" is provisioned |

Send some traffic (a few `/predict` calls) and the dashboard panels fill in: request rate,
p95 latency, predictions by class, status codes, 5xx rate.

## 9. Kubernetes (Minikube)

```bash
minikube start --driver=docker
minikube image build -t heart-disease-api:1.0.0 .     # or: eval $(minikube docker-env) && docker build ...
kubectl apply -f k8s/deployment.yaml
kubectl get pods                                      # expect 2/2 Running
kubectl get svc heart-disease-api
```

Expose the LoadBalancer: run `minikube tunnel` in a separate terminal (the service then gets an
EXTERNAL-IP), or use `minikube service heart-disease-api --url`. Then:

```bash
curl http://<URL>/health
curl -X POST http://<URL>/predict -H "Content-Type: application/json" -d '{...}'
```

**Access instructions for graders (local deployment):** the API is deployed on a local Minikube
cluster; follow the commands above, or see `screenshots/` and the demo video for the verified run.

## 10. CI/CD (GitHub Actions)

`ci.yml` runs on every push / PR:

1. **lint** - `ruff check`
2. **test** - `pytest` (JUnit report uploaded as artifact)
3. **train** (after lint + test) - download data, build cleaned dataset, execute EDA notebook,
   train with MLflow logging, metrics table in the job summary, artifacts uploaded
4. **docker** (after lint + test) - build the image, run it, call `/health`, `/predict`
   (must contain `confidence`) and `/metrics/`, print container logs

Any failing step fails the workflow. `tune.yml` is a manual hyper-parameter-tuning run.
