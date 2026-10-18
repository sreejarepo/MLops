# Heart Disease Prediction: End-to-End MLOps
**Student:** SREEJA V J  
**Course:** MLOps 
**Repository:** https://github.com/sreejarepo/MLops
**Date:** 18-10-2026 
**Demo video:** [link]

> Educational project. The model is **not** a medical diagnostic device.

## 1. Abstract and objectives
Binary classifier predicting heart-disease risk from 13 UCI Heart Disease features,
delivered as a cloud-ready, monitored API. Objectives: reproducible training,
experiment tracking (MLflow), automated CI/CD (GitHub Actions), containerization
(Docker), Kubernetes deployment, and Prometheus/Grafana monitoring.

## 2. Dataset and acquisition
UCI Heart Disease dataset, **ID 45** (https://archive.ics.uci.edu/dataset/45/heart+disease),
fetched programmatically by `download_data.py` via the official `ucimlrepo` API
(no manual downloads). [FILL: rows x features after your run; e.g., 303 rows x 13 features].
The multi-class target (0–4) is binarized: 0 = no disease, >0 = disease present.
License/attribution: cite UCI ML Repository and the original donors (Cleveland Clinic Foundation).

## 3. Data quality and EDA
[FILL: missing-value summary from notebook §3 — e.g., "no missing values after cleaning, 0 duplicates".]
![Class balance](screenshots/eda_class_balance.png)
![Histograms](screenshots/eda_histograms.png)
![Correlation heatmap](screenshots/eda_heatmap.png)
Key observations: [FILL 3–4 bullets, e.g. thalach negative correlation, cp/exang positive,
roughly balanced classes, skewed continuous features justifying scaling.]

## 4. Preprocessing and feature engineering
- Median imputation for numeric features; most-frequent imputation for categorical codes.
- `StandardScaler` on numerics; `OneHotEncoder(handle_unknown="ignore")` on categorical codes.
- All preprocessing lives **inside** the sklearn Pipeline (fit on train folds only), preventing leakage.
- Stratified train/test split (80/20, seed 42) preserves class ratio.

## 5. Model development, tuning, and evaluation
Baselines: Logistic Regression (class_weight=balanced, max_iter=2000) and
Random Forest (300 trees, min_samples_leaf=2, class_weight=balanced).
Selection metric: **mean 5-fold stratified CV ROC-AUC** (test set never used for selection).
Tuning: `GridSearchCV` over C/solver (LR) and n_estimators/max_depth/min_samples_leaf/
max_features (RF); best params logged in MLflow experiment `heart-disease-tuning`;
a tuned pipeline is promoted only if it beats the baseline CV score.

| Model | CV ROC-AUC | Test Acc | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Random Forest | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Tuned winner | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

![Confusion matrix](screenshots/model_confusion_matrix.png)
![ROC curve](screenshots/model_roc_curve.png)
Chosen model: [FILL], because [FILL — e.g., highest CV ROC-AUC with simpler interpretation].

## 6. MLflow experiment tracking
Every run logs: params (model name, test size, seed, CV folds, tuned hyperparameters),
metrics (5 CV scores + 5 held-out test scores), and artifacts (confusion matrix,
ROC curve, classification report, tuned model). Backend: sqlite:///mlflow.db (file store ./mlruns).
![MLflow experiments](screenshots/mlflow_experiments.png)
![MLflow run detail](screenshots/mlflow_run_detail.png)
![MLflow artifacts](screenshots/mlflow_artifacts.png)

## 7. Packaging and reproducibility
Final pipeline saved with joblib to `models/heart_disease_model.joblib` (preprocessing included).
Pinned `requirements.txt`, fixed random seeds. Reproduce from scratch:
`pip install -r requirements.txt && python download_data.py && python -m src.train && python -m src.tune`

## 8. Automated tests and CI/CD
Pytest suites: `tests/test_data.py` (loading/validation), `tests/test_model.py`
(preprocessing + training contracts), `tests/test_api.py` (endpoints, schemas).
GitHub Actions (`ci.yml`) on every push/PR: Ruff lint -> pytest -> dataset download ->
EDA notebook execution -> training with MLflow logging -> artifact upload.
A manual `tune.yml` workflow runs GridSearchCV on demand. The pipeline fails on any
lint/test error, and `if: always()` guarantees artifact collection.
![CI green run](screenshots/ci_pipeline_green.png)
![CI artifacts](screenshots/ci_artifacts.png)

## 9. API, containerization, deployment
FastAPI service (`src/api.py`): `POST /predict` (13 features -> prediction + class
probabilities), `GET /health` (K8s probes), `GET /metrics` (Prometheus client),
request logging middleware. Docker image `heart-disease-api:1.0.0` built from a
python:3.11-slim Dockerfile; verified locally with `docker build/run` and a live
/predict curl.
![Docker build](screenshots/docker_build.png)
![Docker predict](screenshots/docker_predict.png)
Kubernetes (`k8s/deployment.yaml`): 2-replica Deployment with readiness/liveness
probes and resource limits, exposed via LoadBalancer Service on [Minikube / GKE / EKS / AKS].
![kubectl pods](screenshots/kubectl_pods.png)
![kubectl service](screenshots/kubectl_svc.png)
![K8s predict](screenshots/k8s_predict.png)
Access: [FILL: public URL, or `minikube service heart-disease-api` instructions].

## 10. Monitoring, limitations, conclusion
Prometheus scrapes `/metrics` (request counter + latency histogram per endpoint);
Grafana dashboards visualize traffic. Container logs show per-request JSON log lines.
![Prometheus targets](screenshots/prometheus_targets.png)
![Grafana dashboard](screenshots/grafana_dashboard.png)
Limitations: small single-site dataset (Cleveland), no external/clinical validation,
class-associations are not causal, model must not be used for diagnosis.
Future work: drift detection, shadow deployment, ONNX export, larger multi-site data.

## Appendix: Architecture
![Architecture](screenshots/architecture_diagram.png)
