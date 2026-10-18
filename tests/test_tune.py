import json

from src.preprocessing import make_models
from src.tune import PARAM_GRIDS, run_tuning

TINY_GRIDS = {
    "logistic_regression": {"classifier__C": [0.1, 1.0]},
    "random_forest": {"classifier__n_estimators": [10], "classifier__max_depth": [3]},
}


def _baseline(report_dir, score):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_summary.json").write_text(json.dumps({
        "selected_model": "logistic_regression", "selection_score": score,
        "models": [{"model": "logistic_regression", "cv_roc_auc_mean": score},
                   {"model": "random_forest", "cv_roc_auc_mean": score}]}))


def test_param_grids_match_pipelines(sample_df):
    X = sample_df.drop(columns="target")
    assert set(PARAM_GRIDS) == set(make_models(X))


def test_every_grid_parameter_exists_on_its_pipeline(sample_df):
    X = sample_df.drop(columns="target")
    for name, model in make_models(X).items():
        valid = model.get_params()
        for param in PARAM_GRIDS[name]:
            assert param in valid, f"{param} is not a parameter of the {name} pipeline"


def test_tuning_promotes_when_it_beats_baseline(sample_csv, tmp_path):
    report_dir, model_path = tmp_path / "reports", tmp_path / "model.joblib"
    _baseline(report_dir, 0.0)
    summary = run_tuning(sample_csv, model_path=model_path, report_dir=report_dir,
                         param_grids=TINY_GRIDS)
    assert summary["promoted_tuned_model"] is True
    assert model_path.exists()
    final = json.loads((report_dir / "final_model.json").read_text())
    assert final["source"] == "tuned"
    assert (report_dir / "plots" / "tuned_random_forest_roc_curve.png").exists()
    assert (report_dir / "tuned_logistic_regression_cv_results.csv").exists()


def test_tuning_does_not_overwrite_when_baseline_is_better(sample_csv, tmp_path):
    report_dir, model_path = tmp_path / "reports", tmp_path / "model.joblib"
    _baseline(report_dir, 1.01)  # unbeatable
    summary = run_tuning(sample_csv, model_path=model_path, report_dir=report_dir,
                         param_grids=TINY_GRIDS)
    assert summary["promoted_tuned_model"] is False
    assert not model_path.exists()


def test_no_promote_flag_never_writes_model(sample_csv, tmp_path):
    report_dir, model_path = tmp_path / "reports", tmp_path / "model.joblib"
    _baseline(report_dir, 0.0)
    summary = run_tuning(sample_csv, refit_model=False, model_path=model_path,
                         report_dir=report_dir, param_grids=TINY_GRIDS)
    assert summary["promoted_tuned_model"] is False
    assert not model_path.exists()
