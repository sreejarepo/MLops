"""Download the UCI Heart Disease dataset using the official UCI ML Repository API."""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Normalize column names and common missing-value placeholders.
    X.columns = [str(c).strip().lower().replace(" ", "_") for c in X.columns]
    y.columns = [str(c).strip().lower().replace(" ", "_") for c in y.columns]
    df = X.copy()
    target_col = y.columns[0]
    df["target"] = y[target_col]
    df = df.replace(["?", " ?", "", "NA", "N/A"], pd.NA)

    # UCI processed Cleveland data commonly encodes target as 0..4.
    # Binary target: 0 = no disease; 1..4 = disease present.
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"])
    df["target"] = (df["target"] > 0).astype(int)
    out = RAW_DIR / "heart_disease.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows and {len(df.columns)-1} features to {out}")
    print("Columns:", ", ".join(df.columns))

if __name__ == "__main__":
    main()
