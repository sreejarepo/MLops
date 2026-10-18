"""Shared fixtures: a small synthetic heart-disease-like dataset (same 13
feature columns as the UCI data, a few missing values) so tests run offline
and in seconds, and an isolated MLflow backend so tests never touch ./mlflow.db."""
import numpy as np
import pandas as pd
import pytest

PAYLOAD = {"age": 58, "sex": 1, "cp": 3, "trestbps": 140, "chol": 240, "fbs": 0,
           "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0,
           "slope": 2, "ca": 0, "thal": 2}


def make_sample_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    target = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(target)
    df = pd.DataFrame({
        "age": rng.normal(54, 9, n).round(),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.normal(131, 17, n).round(),
        "chol": rng.normal(246, 50, n).round(),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": (150 - 20 * target + rng.normal(0, 15, n)).round(),
        "exang": rng.integers(0, 2, n),
        "oldpeak": (target + rng.normal(0.5, 0.5, n)).clip(0).round(1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "target": target,
    })
    df.loc[[3, 17], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df


@pytest.fixture
def payload():
    return dict(PAYLOAD)


@pytest.fixture
def sample_df():
    return make_sample_df()


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def isolated_mlflow(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlflow-artifacts"))
