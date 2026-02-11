"""
Add derived columns to predictions/test_predictions.csv and write the
result to predictions_combined/STL_ChemProp_predictions.csv.

New columns added:
    Task             – Strain + S9 status, e.g. 'TA97_with_S9' or 'TA97_without_S9'
    Ground Truth     – Endpoint mapped to binary (Positive=1, Negative=0)
    Binary Prediction – prediction cast to int
"""
from pathlib import Path

import pandas as pd


def build_task_column(df: pd.DataFrame) -> pd.Series:
    """Build Task column from Strain + S9: e.g. 'TA97_with_S9' or 'TA97_without_S9'."""
    return df["Strain"] + df["S9"].map({1: "_with_S9", 0: "_without_S9"})


def build_ground_truth(df: pd.DataFrame) -> pd.Series:
    """Map Endpoint to binary ground truth: Positive=1, Negative=0."""
    return df["Endpoint"].map({"Positive": 1, "Negative": 0})


def main():
    base = Path(__file__).resolve().parent
    pred_path = base / "predictions" / "test_predictions.csv"
    out_dir = base / "cleaned_predictions"
    out_dir.mkdir(exist_ok=True)

    print(f"Reading predictions from {pred_path}")
    df = pd.read_csv(pred_path)
    print(f"  Loaded {len(df)} rows")

    # Add derived columns
    df["Task"] = build_task_column(df)
    df["Ground Truth"] = build_ground_truth(df)
    df["Binary Prediction"] = df["prediction"].astype(int)

    # Keep only the relevant columns in a clean order
    df = df[
        ["gmtamesQSAR_ID", "SMILES", "Task", "Ground Truth", "Binary Prediction", "prediction_prob"]
    ]

    out_path = out_dir / "Encoder_Swap_ChemProp_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.name}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
