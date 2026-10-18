import json

import mlflow
import pandas as pd
import pytest

from src.train import run_training


def test_run_training_end_to_end(sample_csv, tmp_path):
    model_dir, report_dir = tmp_path / "models", tmp_path / "reports"
    summary = run_training(sample_csv, model_dir=model_dir, report_dir=report_dir)

    assert (model_dir / "heart_disease_model.joblib").exists()
    assert summary["selected_model"] in {"logistic_regression", "random_forest"}
    assert len(summary["models"]) == 2
    for m in summary["models"]:
        for key in ("cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean",
                    "cv_roc_auc_mean", "test_accuracy", "test_precision",
                    "test_recall", "test_roc_auc"):
            assert 0.0 <= m[key] <= 1.0

    final = json.loads((report_dir / "final_model.json").read_text())
    assert final["source"] == "baseline"
    for plot in ("class_balance.png", "correlation_heatmap.png", "feature_histograms.png",
                 "random_forest_roc_curve.png", "logistic_regression_confusion_matrix.png"):
        assert (report_dir / "plots" / plot).exists()


def test_run_training_logs_every_run_to_mlflow(sample_csv, tmp_path):
    run_training(sample_csv, model_dir=tmp_path / "m", report_dir=tmp_path / "r")
    runs = mlflow.search_runs(experiment_names=["heart-disease-classification"])
    assert len(runs) == 2
    for column in ("params.model_name", "params.random_state", "metrics.cv_roc_auc_mean",
                   "metrics.test_roc_auc", "metrics.test_recall"):
        assert column in runs.columns
        assert runs[column].notna().all()


def test_run_training_rejects_single_class_data(tmp_path, sample_df):
    path = tmp_path / "one_class.csv"
    sample_df.assign(target=0).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        run_training(path, model_dir=tmp_path / "m", report_dir=tmp_path / "r")


def test_run_training_fails_on_missing_target(tmp_path, sample_df):
    path = tmp_path / "no_target.csv"
    sample_df.drop(columns="target").to_csv(path, index=False)
    with pytest.raises(ValueError):
        run_training(path, model_dir=tmp_path / "m", report_dir=tmp_path / "r")


def test_training_is_reproducible(sample_csv, tmp_path):
    a = run_training(sample_csv, model_dir=tmp_path / "m1", report_dir=tmp_path / "r1")
    b = run_training(sample_csv, model_dir=tmp_path / "m2", report_dir=tmp_path / "r2")
    scores_a = pd.Series({m["model"]: m["cv_roc_auc_mean"] for m in a["models"]})
    scores_b = pd.Series({m["model"]: m["cv_roc_auc_mean"] for m in b["models"]})
    pd.testing.assert_series_equal(scores_a, scores_b)
