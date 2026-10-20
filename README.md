## Important Note

The final end-to-end report is available as [`report/REPORT.pdf`](report/REPORT.pdf) (PDF) and [`report/REPORT.md`](report/REPORT.md) (Markdown).
Please refer to it for all links, steps, screenshots, output values, observations and analysis.


# Heart Disease MLOps

End-to-end MLOps project on the UCI Heart Disease dataset: EDA, model training and
tuning, MLflow experiment tracking, tests + CI/CD, Docker, Kubernetes deployment and
Prometheus/Grafana monitoring. **Educational project - not a clinical diagnostic system.**

**Demo video:** https://drive.google.com/file/d/19tLMqSR8QTFi2KamsZ3QJQGIrb-G8ACh/view?usp=drive_link 
**Report:** `report/REPORT.md` (PDF export in `report/REPORT.pdf`)

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

The image is built on the host with Docker (section 8) and is not in a registry, so it is loaded into
the cluster. `k8s/deployment.yaml` uses `imagePullPolicy: IfNotPresent` for this reason.

```bash
minikube start --driver=docker
minikube image load heart-disease-api:1.0.0
minikube kubectl -- apply -f k8s/deployment.yaml
minikube kubectl -- rollout status deployment/heart-disease-api
minikube kubectl -- get pods,svc                      # expect 2 pods Running 1/1
```

`minikube kubectl --` runs the kubectl that comes with Minikube, so no separate kubectl install is
needed. If you have kubectl installed, use plain `kubectl` instead.

Reach the service. Port-forward is the simplest; leave it running in a separate terminal:

```bash
minikube kubectl -- port-forward svc/heart-disease-api 8080:80
```

Then, from another terminal:

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
minikube kubectl -- logs deploy/heart-disease-api --tail=20
```

Alternative: run `minikube tunnel` in a separate terminal so the LoadBalancer service gets an
EXTERNAL-IP, and call that address on port 80. The cluster state from my run is saved in
`reports/k8s_state.txt`.

**Access instructions for graders (local deployment):** the API is deployed on a local Minikube
cluster. Follow the commands above and call it at `http://127.0.0.1:8080`, or see `screenshots/` and
the demo video for the verified run.

## 10. CI/CD (GitHub Actions)

`ci.yml` runs on every push / PR:

1. **lint** - `ruff check`
2. **test** - `pytest` (JUnit report uploaded as artifact)
3. **train** (after lint + test) - download data, build cleaned dataset, execute EDA notebook,
   train with MLflow logging, metrics table in the job summary, artifacts uploaded
4. **docker** (after lint + test) - build the image, run it, call `/health`, `/predict`
   (must contain `confidence`) and `/metrics/`, print container logs

Any failing step fails the workflow. `tune.yml` is a manual hyper-parameter-tuning run.