# Heart Disease Prediction: End-to-End MLOps
**Student:** [Your name]  
**Course:** [Course name]  
**Repository:** [GitHub URL]  
**Date:** [Submission date]

## 1. Abstract and objectives
Describe the classification task and the MLOps objective. State that the model is educational and not for clinical use.

## 2. Dataset and acquisition
Document UCI dataset ID 45, source URL, selected variant, number of rows/features, target conversion, and data license/attribution as required by UCI.

## 3. Data quality and EDA
Insert class balance, histograms, correlation heatmap, and missing-value summary. Explain notable observations without claiming causation.

## 4. Preprocessing and feature engineering
Explain imputation, categorical encoding, scaling, stratified split, and leakage prevention.

## 5. Model development and evaluation
Describe Logistic Regression and Random Forest, cross-validation, selection criterion, hyperparameters, and held-out test metrics. Insert actual values from `reports/training_summary.json`.

## 6. MLflow experiment tracking
Insert screenshots of experiment runs and artifacts. Explain how parameters, metrics, and plots were logged.

## 7. Packaging and reproducibility
Describe saved `joblib` pipeline, requirements file, random seed, and commands to reproduce training.

## 8. Automated tests and CI/CD
Describe unit tests and the GitHub Actions workflow. Insert passing workflow screenshot and artifact details.

## 9. API, containerization, deployment
Explain `/predict`, `/health`, `/metrics`, Docker image, Kubernetes Deployment and Service. Include genuine test/deployment screenshots and endpoint/access instructions.

## 10. Monitoring, limitations, conclusion
Describe API request/latency metrics, logs, dashboard, limitations, bias/data representativeness, lack of clinical validation, and future work.
