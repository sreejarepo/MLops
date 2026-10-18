"""Produce the cleaned dataset deliverable (data/processed/heart_disease_clean.csv)
plus a small data-quality report.

Note: model training still starts from the RAW csv and imputes *inside* the
sklearn pipeline (fit on training folds only) to avoid leakage. This script
exists to give a human-inspectable, fully cleaned copy of the data.

    python -m src.clean
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import TARGET, load_dataset
from src.preprocessing import CATEGORICAL_CANDIDATES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def clean_dataset(df: pd.DataFrame):
    """Return (clean_df, quality_report). Median-imputes numeric columns,
    mode-imputes coded categoricals, drops duplicates, fixes dtypes."""
    df = df.copy()
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical and c != TARGET]

    for col in numeric + categorical:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report = {
        "rows_in": int(len(df)),
        "duplicates_dropped": int(df.duplicated().sum()),
        "missing_before": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "imputation": {},
    }
    df = df.drop_duplicates().reset_index(drop=True)

    for col in numeric:
        if df[col].isna().any():
            value = float(df[col].median())
            df[col] = df[col].fillna(value)
            report["imputation"][col] = f"median={value}"
    for col in categorical:
        if df[col].isna().any():
            value = float(df[col].mode().iloc[0])
            df[col] = df[col].fillna(value)
            report["imputation"][col] = f"mode={value}"
        df[col] = df[col].astype(int)
    df[TARGET] = df[TARGET].astype(int)

    report["rows_out"] = int(len(df))
    report["missing_after"] = int(df.isna().sum().sum())
    report["class_counts"] = {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()}
    return df, report


def main(raw_path=RAW, out_dir=PROCESSED_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean, report = clean_dataset(load_dataset(raw_path))
    csv_path = out_dir / "heart_disease_clean.csv"
    clean.to_csv(csv_path, index=False)
    (out_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved cleaned dataset -> {csv_path}")
    return clean, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()
    main(args.raw, args.out_dir)
