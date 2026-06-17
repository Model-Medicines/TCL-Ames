"""
Combine per-task STL GROVER prediction CSVs into a single summary CSV.

Pretrained GROVER predictions live under:
    predictions_pretrained/<TaskName>_predictions.csv
Each file has columns:
    (blank), ames_mutagenicity
where the first column is SMILES and ames_mutagenicity is prediction probability.

Ground truth labels are read from:
    data_processed/test/<TaskName>.csv
with columns: smiles, ames_mutagenicity

Output summary CSV columns:
    SMILES, Task, Ground Truth, Binary Prediction, prediction_prob
"""
from pathlib import Path

import pandas as pd


def load_ground_truth(test_csv_path: Path) -> pd.DataFrame:
    """Read task test labels and return standard columns: SMILES, Ground Truth."""
    labels = pd.read_csv(test_csv_path)
    labels = labels.rename(columns={"smiles": "SMILES", "ames_mutagenicity": "Ground Truth"})
    return labels[["SMILES", "Ground Truth"]]


def load_grover_prediction(pred_csv_path: Path, task_name: str, test_csv_path: Path) -> pd.DataFrame:
    """Read one pretrained GROVER prediction CSV and return a standardised DataFrame."""
    pred_df = pd.read_csv(pred_csv_path)
    first_col = pred_df.columns[0]  # unnamed SMILES column
    pred_df = pred_df.rename(columns={first_col: "SMILES", "ames_mutagenicity": "prediction_prob"})

    labels_df = load_ground_truth(test_csv_path)
    if len(pred_df) == len(labels_df) and pred_df["SMILES"].equals(labels_df["SMILES"]):
        pred_df["Ground Truth"] = labels_df["Ground Truth"].astype(int).values
    else:
        # Fallback to key-based merge if row order differs.
        pred_df = pred_df.merge(labels_df, on="SMILES", how="left")
        if pred_df["Ground Truth"].isna().any():
            raise ValueError(f"Missing ground truth labels after merge for task {task_name}")
        pred_df["Ground Truth"] = pred_df["Ground Truth"].astype(int)

    pred_df["Task"] = task_name
    pred_df["Binary Prediction"] = (pred_df["prediction_prob"] >= 0.5).astype(int)
    return pred_df


def load_predictions(pred_dir: Path, test_dir: Path) -> pd.DataFrame:
    """Read all pretrained task prediction CSVs and concatenate them."""
    pred_files = sorted(pred_dir.glob("*_predictions.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No *_predictions.csv files found in {pred_dir}")

    dfs = []
    for pred_csv_path in pred_files:
        task_name = pred_csv_path.stem.replace("_predictions", "")
        test_csv_path = test_dir / f"{task_name}.csv"
        if not test_csv_path.exists():
            print(f"  WARNING: {test_csv_path} not found - skipping")
            continue
        df = load_grover_prediction(pred_csv_path, task_name=task_name, test_csv_path=test_csv_path)
        dfs.append(df)
        print(f"  Loaded {pred_csv_path.name}  ({len(df)} rows)")

    if not dfs:
        raise FileNotFoundError(
            f"No valid pretrained prediction files found in {pred_dir} with matching labels in {test_dir}"
        )

    combined = pd.concat(dfs, ignore_index=True)
    # Keep only the relevant columns in a clean order
    combined = combined[
        ["SMILES", "Task", "Ground Truth", "Binary Prediction", "prediction_prob"]
    ]
    return combined


def main():
    base = Path(__file__).resolve().parent
    pred_dir = base / "predictions_pretrained"
    test_dir = base / "data_processed" / "test"
    out_dir = base / "predictions_combined"
    out_dir.mkdir(exist_ok=True)

    print(f"Reading STL GROVER predictions from {pred_dir}/")
    combined = load_predictions(pred_dir, test_dir)

    out_path = out_dir / "STL_GROVER_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote combined predictions to {out_path.name}  ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
