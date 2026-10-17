import pandas as pd
import pytest

from src.data import load_dataset, split_features_target


def test_load_dataset_converts_target_and_missing(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"age": [50, 60, 55], "chol": [200, "?", 240],
                  "target": [0, 1, 2]}).to_csv(path, index=False)
    df = load_dataset(path)
    assert df["target"].tolist() == [0, 1, 1]
    assert pd.isna(df.loc[1, "chol"])

def test_load_dataset_requires_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [50]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)

def test_split_features_target():
    X, y = split_features_target(pd.DataFrame({"age": [40, 50], "target": [0, 1]}))
    assert list(X.columns) == ["age"]
    assert y.tolist() == [0, 1]
