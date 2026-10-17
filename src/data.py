"""Data loading and validation helpers."""
import pandas as pd

TARGET = "target"

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    df = df.replace(["?", " ?", "", "NA", "N/A"], pd.NA)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df

def split_features_target(df):
    if "target" not in df.columns:
        raise ValueError("Missing target column.")
    return df.drop(columns=["target"]), df["target"].astype(int)
