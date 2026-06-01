"""
Score new customers with the trained churn Pipeline.

Run from the repo root (after train.py):

    python scripts/predict.py

Reads `data/new_customers.csv` (same schema as training, no `Churn` column),
sets `customerID` aside, applies the saved Pipeline + frozen threshold, and
writes `results/predictions.csv` with columns: customerID, churn_pred, churn_proba.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

# Importing the module makes the custom transformer available when the pickled
# Pipeline is unpickled (its class is referenced as preprocessing.*).
import preprocessing  # noqa: F401
from preprocessing import ID_COL, TARGET

ROOT = Path(__file__).resolve().parents[1]
NEW_CUSTOMERS = ROOT / "data" / "new_customers.csv"
PIPE_PATH = ROOT / "results" / "churn_pipeline.pkl"
THR_PATH = ROOT / "results" / "threshold.json"
OUT_PATH = ROOT / "results" / "predictions.csv"


def main():
    pipe = joblib.load(PIPE_PATH)
    threshold = json.loads(THR_PATH.read_text())["threshold"]

    df = pd.read_csv(NEW_CUSTOMERS)
    ids = df[ID_COL]
    # Drop identifier (not a feature) and target if it happens to be present.
    features = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])

    proba = pipe.predict_proba(features)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        ID_COL: ids,
        "churn_pred": pred,
        "churn_proba": proba.round(4),
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"Scored {len(out)} customers at threshold {threshold:.3f}")
    print(out.to_string(index=False))
    print(f"\nSaved -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
