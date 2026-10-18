# Heart Disease MLOps

Educational end-to-end MLOps project: UCI Heart Disease classifier with experiment tracking,
CI/CD, Docker containerization, Kubernetes deployment, and monitoring.
**Not a clinical diagnostic system.**

## 1. Setup

Python 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Download data

```bash
python download_data.py
```

Fetches UCI dataset ID 45 via `ucimlrepo`, normalizes the target to binary
(0 = no disease; >0 = disease present), and saves `data/raw/heart_disease.csv`.

## 3. EDA

Open and run `notebooks/01_eda.ipynb` (missing values, class balance,
histograms, correlation heatmap, feature-vs-target boxplots).

## 4. Train, tune, and track experiments

```bash
python -m src.train     # baselines + CV, logs to MLflow, saves reports/plots
python -m src.tune      # GridSearchCV tuning, MLflow experiment "heart-disease-tuning"
```

MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# or, when using the default file store:
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000.

## 5. Tests and linting

```bash
ruff check src tests download_data.py
pytest -q
```

## 6. Run the API locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- Interactive docs: http://localhost:8000/docs
- Health: `GET http://localhost:8000/health`
- Metrics: `GET http://localhost:8000/metrics`
- Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":3,"trestbps":140,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

## 7. Batch inference

```bash
python -m src.inference --sample
python -m src.inference --input my_patients.csv --output predictions.csv
```

## 8. Docker

```bash
docker build -t heart-disease-api:1.0.0 .
docker run -p 8000:8000 heart-disease-api:1.0.0
```

Or API + Prometheus + Grafana together:

```bash
docker compose up --build
# API on :8002, Prometheus on :9090, Grafana on :3000
```

## 9. Kubernetes (Minikube)

```bash
minikube start
eval $(minikube docker-env)          # Linux/macOS; on Windows: minikube docker-env --shell powershell | Invoke-Expression
docker build -t heart-disease-api:1.0.0 .
kubectl apply -f k8s/deployment.yaml
kubectl get pods                     # expect 2/2 Running
minikube service heart-disease-api   # get the LoadBalancer URL
curl -X POST <URL>/predict -H "Content-Type: application/json" -d '{...}'
```

## 10. CI/CD

`.github/workflows/ci.yml` runs on every push/PR: Ruff lint -> pytest -> data
download -> training with MLflow logging -> artifact upload.
`.github/workflows/tune.yml` is a manual workflow_dispatch for hyperparameter
tuning.
