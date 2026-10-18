import numpy as np
import pandas as pd

from src.preprocessing import make_models, make_preprocessor


def test_models_fit_and_predict():
    X = pd.DataFrame({
        "age": [40, 50, 60, 35, 65, 45, 55, 70, 42, 62, 38, 58],
        "sex": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "cp": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        "chol": [180, 210, 240, 190, 260, 200, 230, 250, 185, 255, 195, 225],
    })
    y = pd.Series([0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    for model in make_models(X).values():
        model.fit(X, y)
        assert len(model.predict(X)) == len(y)
        assert model.predict_proba(X).shape == (len(y), 2)


def test_pipeline_handles_missing_values_and_unseen_categories(sample_df):
    X, y = sample_df.drop(columns="target"), sample_df["target"]
    new = X.head(3).copy()
    new.loc[new.index[0], "ca"] = np.nan   # missing value at inference time
    new.loc[new.index[1], "cp"] = 9        # category never seen in training
    for model in make_models(X).values():
        model.fit(X, y)
        proba = model.predict_proba(new)
        assert proba.shape == (3, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_scaler_is_fit_on_training_data_only(sample_df):
    X = sample_df.drop(columns="target")
    X_train = X.iloc[:60]
    pre = make_preprocessor(X_train).fit(X_train)
    scaler = pre.named_transformers_["numeric"].named_steps["scaler"]
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    assert np.allclose(scaler.mean_, X_train[numeric].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X[numeric].mean().to_numpy())


def test_model_names_are_stable(sample_df):
    X = sample_df.drop(columns="target")
    assert set(make_models(X)) == {"logistic_regression", "random_forest"}
