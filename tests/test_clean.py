import json

import pandas as pd

from src.clean import clean_dataset, main


def test_clean_dataset_imputes_and_types(sample_df):
    clean, report = clean_dataset(sample_df)
    assert clean.isna().sum().sum() == 0
    assert report["missing_before"] == {"ca": 2, "thal": 1}
    assert report["missing_after"] == 0
    assert set(report["imputation"]) == {"ca", "thal"}
    assert str(clean["ca"].dtype).startswith("int")
    assert set(clean["target"].unique()) == {0, 1}


def test_clean_dataset_drops_duplicates(sample_df):
    with_dupe = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    clean, report = clean_dataset(with_dupe)
    assert report["duplicates_dropped"] == 1
    assert len(clean) == len(sample_df)


def test_clean_main_writes_outputs(sample_csv, tmp_path):
    out_dir = tmp_path / "processed"
    main(sample_csv, out_dir)
    assert (out_dir / "heart_disease_clean.csv").exists()
    report = json.loads((out_dir / "data_quality_report.json").read_text())
    assert report["rows_out"] == 80
