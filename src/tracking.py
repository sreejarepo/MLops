"""Single place that configures MLflow, so train.py and tune.py always log to
the same backend and a single `mlflow ui` shows every experiment.

Defaults to a local SQLite database (./mlflow.db). Override with the real
environment variables (not .env, nothing loads that file):

    MLFLOW_TRACKING_URI   e.g. sqlite:///mlflow.db or http://mlflow-server:5000
    MLFLOW_ARTIFACT_ROOT  folder for artifacts when using a sqlite backend
"""
import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]

# Needed so the skops-serialised sklearn pipeline can be logged/loaded.
SKOPS_TRUSTED_TYPES = ["numpy.dtype"]


def default_tracking_uri() -> str:
    return f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def setup_mlflow(experiment_name: str) -> str:
    """Point MLflow at the configured backend and select (or create) the experiment."""
    uri = os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri()
    mlflow.set_tracking_uri(uri)

    # With a sqlite backend MLflow would otherwise drop artifacts into a
    # cwd-relative ./mlruns, so pin an explicit, absolute location instead.
    if uri.startswith("sqlite") and mlflow.get_experiment_by_name(experiment_name) is None:
        root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT") or ROOT / "mlruns" / "artifacts")
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(root / experiment_name).resolve().as_uri(),
        )
    mlflow.set_experiment(experiment_name)
    return uri
