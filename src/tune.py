"""Hyper-parameter tuning with GridSearchCV, tracked in MLflow.

Tunes the Logistic Regression and Random Forest pipelines from
src.preprocessing, compares tuned vs baseline CV ROC-AUC and -- only if tuning
beats the best baseline -- replaces models/heart_disease_model.joblib with the
tuned pipeline. Run after `python -m src.train`.
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data import load_dataset, split_features_target
from src.evaluation import compute_test_metrics, save_eval_artifacts
from src.preprocessing import make_models
from src.tracking import SKOPS_TRUSTED_TYPES, setup_mlflow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"
REPORT_DIR = ROOT / "reports"

# Grids target the classifier step inside each Pipeline ("classifier__" prefix).
# (`penalty` is intentionally not tuned: L2 is the default and the parameter is
# deprecated in recent scikit-learn releases.)
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 6, 12],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    },
}


def load_baseline_summary(report_dir=REPORT_DIR):
    path = Path(report_dir) / "training_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def run_tuning(data_path=DEFAULT_DATA, test_size=0.2, random_state=42, refit_model=True,
               model_path=MODEL_PATH, report_dir=REPORT_DIR, param_grids=None):
    param_grids = param_grids or PARAM_GRIDS
    model_path, report_dir = Path(model_path), Path(report_dir)
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    setup_mlflow("heart-disease-tuning")

    baseline = load_baseline_summary(report_dir)
    baseline_scores = {m["model"]: m.get("cv_roc_auc_mean")
                       for m in (baseline or {}).get("models", [])}

    tuning_results, best_estimators = [], {}
    for name, model in make_models(X_train, random_state=random_state).items():
        with mlflow.start_run(run_name=f"tuned_{name}"):
            grid = GridSearchCV(
                estimator=model, param_grid=param_grids[name], scoring="roc_auc",
                cv=cv, n_jobs=-1, refit=True, error_score="raise",
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            best_estimators[name] = best
            pred = best.predict(X_test)
            proba = best.predict_proba(X_test)[:, 1]
            baseline_cv = baseline_scores.get(name)
            metrics = {
                "tuned_cv_roc_auc": float(grid.best_score_),
                "baseline_cv_roc_auc": -1.0 if baseline_cv is None else float(baseline_cv),
                **compute_test_metrics(y_test, pred, proba),
            }

            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("cv_folds", cv.get_n_splits())
            mlflow.log_param("grid_size", len(grid.cv_results_["params"]))
            mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})
            mlflow.log_metrics(metrics)

            # Plots + report + the full grid so the tuning process is documented.
            for artifact in save_eval_artifacts(f"tuned_{name}", y_test, pred, proba,
                                                plot_dir, report_dir):
                mlflow.log_artifact(str(artifact))
            cv_results_path = report_dir / f"tuned_{name}_cv_results.csv"
            pd.DataFrame(grid.cv_results_).to_csv(cv_results_path, index=False)
            mlflow.log_artifact(str(cv_results_path))
            mlflow.sklearn.log_model(sk_model=best, name="model",
                                     skops_trusted_types=SKOPS_TRUSTED_TYPES)

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
        joblib.dump(best_estimators[winner_name], model_path)
        (report_dir / "final_model.json").write_text(json.dumps({
            "final_model": winner_name, "source": "tuned",
            "cv_roc_auc_mean": best_tuned["tuned_cv_roc_auc"],
            "best_params": best_tuned["best_params"], "model_file": model_path.name,
        }, indent=2))
        promoted = True
        print(f"Promoted tuned {winner_name} -> {model_path}")

    summary = {"promoted_tuned_model": promoted,
               "baseline_selection_score": baseline_best,
               "tuning": tuning_results}
    (report_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
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
