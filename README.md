# Heart Disease MLOps

Educational end-to-end classifier using the UCI Heart Disease dataset. This project is not a clinical diagnostic system.

## 1. Environment setup

Use Python 3.11 in a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
----C:\Users\Sreeja\PycharmProjects\DNN\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Download and inspect data

```bash
python download_data.py
```

The script uses the `ucimlrepo` package to fetch UCI dataset ID 45, normalizes the target to binary (`0` means no disease; values above `0` mean disease present), and saves `data/raw/heart_disease.csv`.

## 3. Train and track experiments
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```bash
python -m src.train
```

This trains Logistic Regression and Random Forest, runs stratified 5-fold cross-validation, evaluates the held-out test set, logs parameters/metrics/plots to MLflow, and saves `models/heart_disease_model.joblib`.

View the MLflow UI in another terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open http://127.0.0.1:5000.

## 4. Run tests and lint

```bash
ruff check src tests download_data.py
pytest -q
```
---fix python -m ruff check --fix src tests download_data.py
--- python -m pytest -q

## 5. Start the API locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
-----http://127.0.0.1:8000/docs
- Health: `GET http://localhost:8000/health`
- Prediction: `POST http://localhost:8000/predict`
- Metrics: `GET http://localhost:8000/metrics`
- Interactive API docs: `http://localhost:8000/docs`

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":3,"trestbps":140,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":3,"trestbps":140,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
---PUT IN PREDICT JSON -----The response will appear under Server response, showing the prediction and class probabilities if the model is available.

Input feature names and valid categories must match the dataset version downloaded.

## 6. Docker

Train the model first. Then:

```bash
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0
```
Alternatively use `docker compose up --build`.

http://127.0.0.1:8000/docs
http://127.0.0.1:8000/metrics/

 mlflow ui --backend-store-uri ./mlruns --port 5000   

## 7. Kubernetes / Minikube

Build the image in Minikube's Docker daemon:

```bash
minikube start -----minikube start --driver=docker ---kubectl get nodes-----sttaus should be ready
----Test-Path .\models\heart_disease_model.joblib 
----minikube image build -t heart-disease-api:1.0.0 .
-----Get-Content .\k8s\deployment.yaml
------kubectl apply -f k8s/deployment.yaml
---kubectl get pods
----kubectl get svc heart-disease-api
---kubectl describe pods
minikube docker-env
# On PowerShell, follow the command printed by `minikube docker-env`.
docker build -t heart-disease-api:1.0.0 .
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get svc heart-disease-api
```

For local Minikube, `LoadBalancer` may need `minikube tunnel` in a separate terminal. A simple alternative for testing is:

```bash
kubectl port-forward svc/heart-disease-api 8000:80 -----kubectl port-forward svc/heart-disease-api 8001:80
```
---kubectl describe pods
Then test `http://localhost:8000/health` and `/docs`.

## 8. Monitoring

The API exposes Prometheus metrics at `/metrics`, including request counts and latency. 
For local Prometheus running in Docker, `host.docker.internal:8000` is used in `monitoring/prometheus.yml`; 
adjust the target to match your environment. Configure Grafana to use Prometheus as a data source and create panels for `api_requests_total` and `api_request_duration_seconds`.
---docker compose config----docker compose up -d --build-----docker compose ps

http://127.0.0.1:9090
http://127.0.0.1:9090/targets
--up
--api_requests_total
--scrape_samples_scraped
## 9. Screenshots and report


Save genuine screenshots under `screenshots/`: EDA plots, MLflow runs, passing tests, GitHub Actions, Docker container, `/predict` response, Kubernetes pods/service, and Grafana panels. Write the report from actual measured results; do not claim cloud deployment unless it was completed.

## Important notes

- A 5-fold CV is used for model selection; the test split is held out for final evaluation.
- Preprocessing is inside the fitted pipeline to avoid train/test leakage.
- The class probabilities are model outputs and are not guaranteed to be calibrated medical risk estimates.
- The GitHub Actions workflow downloads data from UCI, so the hosted runner needs internet access.
