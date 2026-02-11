"""
Combine per-task STL prediction CSVs from predictions/ into a single summary CSV.
Each input CSV has columns:
    gmtamesQSAR_ID, SMILES, Strain, S9, Endpoint, prediction_prob, prediction, prediction_label

Output summary CSV columns:
    gmtamesQSAR_ID, SMILES, Task, Ground Truth, Binary Prediction, prediction_prob
"""
from pathlib import Path

import pandas as pd


def build_task_column(df: pd.DataFrame) -> pd.Series:
    """Build Task column from Strain + S9: e.g. 'TA97_with_S9' or 'TA97_without_S9'."""
    return df["Strain"] + df["S9"].map({1: "_with_S9", 0: "_without_S9"})


def build_ground_truth(df: pd.DataFrame) -> pd.Series:
    """Map Endpoint to binary ground truth: Positive=1, Negative=0."""
    return df["Endpoint"].map({"Positive": 1, "Negative": 0})


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    """Read all *_predictions.csv files from pred_dir, add derived columns, and concatenate."""
    csv_files = sorted(pred_dir.glob("*_predictions.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_predictions.csv files found in {pred_dir}")

    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["Task"] = build_task_column(df)
        df["Ground Truth"] = build_ground_truth(df)
        df["Binary Prediction"] = df["prediction"].astype(int)
        dfs.append(df)
        print(f"  Loaded {csv_path.name}  ({len(df)} rows)")

    combined = pd.concat(dfs, ignore_index=True)
    # Keep only the relevant columns in a clean order
    combined = combined[
        ["gmtamesQSAR_ID", "SMILES", "Task", "Ground Truth", "Binary Prediction", "prediction_prob"]
    ]
    combined = combined.rename(columns={"prediction_prob": "Prediction Probability"})
    return combined


def main():
    base = Path(__file__).resolve().parent
    pred_dir = base / "predictions"
    out_dir = base / "predictions_combined"
    out_dir.mkdir(exist_ok=True)

    print(f"Reading STL predictions from {pred_dir}/")
    combined = load_predictions(pred_dir)

    out_path = out_dir / "STL_ChemProp_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote combined predictions to {out_path.name}  ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
