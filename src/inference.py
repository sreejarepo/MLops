"""Batch inference script — scores a CSV of patient features with the saved model.

Usage:
    python -m src.inference --input data/sample_patients.csv --output predictions.csv
    python -m src.inference --sample            # score a built-in example
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"

SAMPLE_PATIENT = {
    "age": 58, "sex": 1, "cp": 3, "trestbps": 140, "chol": 240, "fbs": 0,
    "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0,
    "slope": 2, "ca": 0, "thal": 2,
}

def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)

def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = model.predict(frame).astype(int)
    proba = model.predict_proba(frame)
    classes = list(model.classes_)
    out["p_disease"] = proba[:, classes.index(1)] if 1 in classes else proba[:, 0]
    out["p_no_disease"] = 1.0 - out["p_disease"]
    out["prediction_label"] = out["prediction"].map({0: "no_disease", 1: "disease"})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV with the 13 feature columns (no target).")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--sample", action="store_true", help="score the built-in sample patient")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    model = load_model(Path(args.model))
    if args.sample:
        frame = pd.DataFrame([SAMPLE_PATIENT])
    elif args.input:
        frame = pd.read_csv(args.input)
    else:
        parser.error("Provide --input <csv> or --sample")

    result = predict_frame(model, frame)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved -> {args.output}")

if __name__ == "__main__":
    main()
