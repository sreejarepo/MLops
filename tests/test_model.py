import pandas as pd

from src.preprocessing import make_models


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
