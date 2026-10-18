"""Train, cross-validate, evaluate and track baseline models with MLflow.

Every run logs: parameters (split/CV settings + classifier hyper-parameters),
5 CV metrics + 5 held-out test metrics, plots (confusion matrix, ROC, EDA),
the classification report, requirements.txt and the fitted pipeline itself.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data import load_dataset, split_features_target
from src.evaluation import compute_test_metrics, save_eval_artifacts
from src.preprocessing import make_models
from src.tracking import SKOPS_TRUSTED_TYPES, setup_mlflow

plt.switch_backend("Agg")  # headless: never try to open a window

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CV_SCORING = {"accuracy": "accuracy", "precision": "precision",
              "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}


def save_eda_plots(df, plot_dir):
    """Class balance, feature histograms and correlation heatmap."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    df.hist(figsize=(14, 10), bins=20, edgecolor="black")
    plt.tight_layout()
    paths.append(plot_dir / "feature_histograms.png")
    plt.savefig(paths[-1], dpi=150)
    plt.close("all")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    paths.append(plot_dir / "correlation_heatmap.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="target", ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no disease (0)", "disease (1)"])
    ax.set_title("Target class balance")
    for patch in ax.patches:
        ax.annotate(f"{int(patch.get_height())}",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    paths.append(plot_dir / "class_balance.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def run_training(data_path=DEFAULT_DATA, test_size=0.2, random_state=42,
                 model_dir=MODEL_DIR, report_dir=REPORT_DIR):
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    plot_dir = report_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    X, y = split_features_target(df)
    if y.nunique() != 2:
        raise ValueError("Expected a binary target with both classes represented.")

    eda_paths = save_eda_plots(df, plot_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    setup_mlflow("heart-disease-classification")

    results = []
    best_name, best_model, best_score, best_run_id = None, None, -1.0, None
    for name, model in make_models(X_train, random_state=random_state).items():
        with mlflow.start_run(run_name=name) as run:
            scores = cross_validate(
                model, X_train, y_train, cv=cv, scoring=CV_SCORING,
                n_jobs=1, error_score="raise",
            )
            cv_metrics = {f"cv_{key[len('test_'):]}_mean": float(values.mean())
                          for key, values in scores.items() if key.startswith("test_")}

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            test_metrics = compute_test_metrics(y_test, pred, proba)

            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("cv_folds", cv.get_n_splits())
            mlflow.log_params({f"clf_{k}": v for k, v in
                               model.named_steps["classifier"].get_params().items()})
            mlflow.log_metrics({**cv_metrics, **test_metrics})

            for artifact in save_eval_artifacts(name, y_test, pred, proba, plot_dir, report_dir):
                mlflow.log_artifact(str(artifact))
            for eda_plot in eda_paths:
                mlflow.log_artifact(str(eda_plot), artifact_path="eda")
            mlflow.log_artifact(str(ROOT / "requirements.txt"))
            mlflow.sklearn.log_model(sk_model=model, name="model",
                                     skops_trusted_types=SKOPS_TRUSTED_TYPES)

            # Selection uses CV ROC-AUC only; the test split is never used to choose.
            selection_score = cv_metrics["cv_roc_auc_mean"]
            mlflow.log_metric("selection_cv_roc_auc", selection_score)
            results.append({"model": name, **cv_metrics, **test_metrics,
                            "run_id": run.info.run_id})
            if selection_score > best_score:
                best_name, best_model = name, model
                best_score, best_run_id = selection_score, run.info.run_id

    mlflow.MlflowClient().set_tag(best_run_id, "selected_baseline", "true")

    model_path = model_dir / "heart_disease_model.joblib"
    joblib.dump(best_model, model_path)
    summary = {"selected_model": best_name, "selection_metric": "mean 5-fold CV ROC-AUC",
               "selection_score": best_score, "models": results}
    (report_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    # Single source of truth for "which model is actually deployed".
    (report_dir / "final_model.json").write_text(json.dumps({
        "final_model": best_name, "source": "baseline",
        "cv_roc_auc_mean": best_score, "model_file": model_path.name}, indent=2))
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
