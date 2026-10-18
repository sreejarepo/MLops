"""Hyperparameter tuning with GridSearchCV, tracked in MLflow.

Tunes Logistic Regression and Random Forest pipelines from src.preprocessing,
compares tuned vs baseline CV ROC-AUC, and — only if tuning improves the best
baseline — replaces models/heart_disease_model.joblib with the tuned pipeline.
Run after `python -m src.train`.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data import load_dataset, split_features_target
from src.preprocessing import make_models

# # Configure MLflow to use a SQLite tracking backend
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
# mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"
REPORT_DIR = ROOT / "reports"

# Grids target the classifier step inside each sklearn Pipeline, hence the
# "classifier__" prefix. Kept small so CI/local runs finish quickly.
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 6, 12],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    },
}

def load_baseline_summary():
    path = REPORT_DIR / "training_summary.json"
    return json.loads(path.read_text()) if path.exists() else None

def run_tuning(data_path=DEFAULT_DATA, test_size=0.2, random_state=42, refit_model=True):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    # mlflow.set_experiment("heart-disease-tuning")

    # Use SQLite unless an explicit tracking URI is configured.
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-tuning")

    baseline = load_baseline_summary()
    baseline_scores = {m["model"]: m.get("cv_roc_auc_mean") for m in (baseline or {}).get("models", [])}

    tuning_results = []
    for name, model in make_models(X_train, random_state=random_state).items():
        with mlflow.start_run(run_name=f"tuned_{name}"):
            grid = GridSearchCV(
                estimator=model, param_grid=PARAM_GRIDS[name], scoring="roc_auc",
                cv=cv, n_jobs=-1, refit=True, error_score="raise",
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            pred = best.predict(X_test)
            proba = best.predict_proba(X_test)[:, 1]
            metrics = {
                "tuned_cv_roc_auc": float(grid.best_score_),
                "baseline_cv_roc_auc": float(baseline_scores.get(name) or -1),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_precision": float(precision_score(y_test, pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, proba)),
            }
            mlflow.log_param("model_name", name)
            mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=best,
                name="model",
                skops_trusted_types=["numpy.dtype"]
            )
            tuning_results.append({"model": name, "best_params": grid.best_params_, **metrics})
            print(f"[{name}] best params: {grid.best_params_}")
            print(f"[{name}] tuned CV ROC-AUC={metrics['tuned_cv_roc_auc']:.4f} "
                  f"baseline={metrics['baseline_cv_roc_auc']:.4f}")

    # Promote the tuned pipeline only if it beats the stored baseline model.
    best_tuned = max(tuning_results, key=lambda r: r["tuned_cv_roc_auc"])
    baseline_best = (baseline or {}).get("selection_score", -1)
    promoted = False
    if refit_model and best_tuned["tuned_cv_roc_auc"] > baseline_best:
        winner_name = best_tuned["model"]
        winner = GridSearchCV(
            estimator=make_models(X_train, random_state=random_state)[winner_name],
            param_grid=PARAM_GRIDS[winner_name], scoring="roc_auc",
            cv=cv, n_jobs=-1, refit=True,
        ).fit(X_train, y_train).best_estimator_
        joblib.dump(winner, MODEL_PATH)
        promoted = True
        print(f"Promoted tuned {winner_name} -> {MODEL_PATH}")

    summary = {"promoted_tuned_model": promoted,
               "baseline_selection_score": baseline_best,
               "tuning": tuning_results}
    (REPORT_DIR / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-promote", action="store_true", help="never overwrite the saved model")
    args = parser.parse_args()
    run_tuning(args.data, args.test_size, args.random_state, refit_model=not args.no_promote)
