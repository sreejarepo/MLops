"""Train, cross-validate, evaluate, and track models with MLflow."""
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data import load_dataset, split_features_target
from src.preprocessing import make_models

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = REPORT_DIR / "plots"

def run_training(data_path=DEFAULT_DATA, test_size=0.2, random_state=42):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    X, y = split_features_target(df)
    if y.nunique() != 2:
        raise ValueError("Expected a binary target with both classes represented.")

    # EDA artifacts
    df.hist(figsize=(14, 10), bins=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_histograms.png", dpi=150)
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="target")
    plt.title("Target class balance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_balance.png", dpi=150)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    mlflow.set_experiment("heart-disease-classification")

    results = []
    best_name, best_model, best_score = None, None, -1.0
    models = make_models(X_train, random_state=random_state)
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            scores = cross_validate(
                model, X_train, y_train, cv=cv,
                scoring={"accuracy": "accuracy", "precision": "precision",
                         "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"},
                n_jobs=1, error_score="raise"
            )
            cv_metrics = {f"cv_{metric}_mean": float(values.mean())
                          for metric, values in scores.items() if metric.startswith("test_")
                          for metric in [metric.replace("test_", "")]}
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_precision": float(precision_score(y_test, pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, proba)),
            }
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metrics({**cv_metrics, **metrics})

            cm = confusion_matrix(y_test, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.tight_layout()
            cm_path = PLOT_DIR / f"{name}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close()
            RocCurveDisplay.from_predictions(y_test, proba)
            plt.tight_layout()
            roc_path = PLOT_DIR / f"{name}_roc_curve.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            report_path = REPORT_DIR / f"{name}_classification_report.txt"
            report_path.write_text(classification_report(y_test, pred, zero_division=0))
            for artifact in [cm_path, roc_path, report_path]:
                mlflow.log_artifact(str(artifact))

            # Use mean cross-validated ROC-AUC for selection; test data is not used to select.
            selection_score = cv_metrics["cv_roc_auc_mean"]
            mlflow.log_metric("selection_cv_roc_auc", selection_score)
            results.append({"model": name, **cv_metrics, **metrics, "run_id": run.info.run_id})
            if selection_score > best_score:
                best_name, best_model, best_score = name, model, selection_score
                mlflow.set_tag("selection", "current_best_cv_roc_auc")

    model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(best_model, model_path)
    summary = {"selected_model": best_name, "selection_metric": "mean 5-fold CV ROC-AUC",
               "selection_score": best_score, "models": results}
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved final pipeline: {model_path}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run_training(args.data, args.test_size, args.random_state)
