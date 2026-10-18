import joblib
import numpy as np
import pandas as pd
import pytest

from src import inference
from src.preprocessing import make_models


@pytest.fixture
def model_file(tmp_path, sample_df):
    X, y = sample_df.drop(columns="target"), sample_df["target"]
    path = tmp_path / "model.joblib"
    joblib.dump(make_models(X)["logistic_regression"].fit(X, y), path)
    return path


def test_sample_patient_matches_training_features(sample_df):
    assert set(inference.SAMPLE_PATIENT) == set(sample_df.drop(columns="target").columns)


def test_predict_frame_adds_prediction_columns(model_file):
    model = inference.load_model(model_file)
    result = inference.predict_frame(model, pd.DataFrame([inference.SAMPLE_PATIENT]))
    assert result.loc[0, "prediction"] in (0, 1)
    assert np.isclose(result.loc[0, "p_disease"] + result.loc[0, "p_no_disease"], 1.0)
    assert result.loc[0, "prediction_label"] in ("disease", "no_disease")


def test_load_model_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.load_model(tmp_path / "nope.joblib")


def test_cli_sample_writes_csv(model_file, tmp_path, monkeypatch, capsys):
    out = tmp_path / "pred.csv"
    monkeypatch.setattr("sys.argv", ["inference", "--sample", "--model", str(model_file),
                                     "--output", str(out)])
    inference.main()
    assert out.exists()
    assert "prediction" in pd.read_csv(out).columns
    assert "Saved" in capsys.readouterr().out
