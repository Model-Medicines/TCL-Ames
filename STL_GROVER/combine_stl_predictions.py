"""
Combine per-task STL GROVER prediction CSVs into a single summary CSV.

GROVER results live under results/<TaskName>/fold_0/test_result.csv
Each result CSV has a two-row header:
    Row 0:  (blank), preds,                targets
    Row 1:  (blank), ames_mutagenicity,    ames_mutagenicity
    Data :  SMILES,  prediction_prob,       ground_truth

Output summary CSV columns:
    SMILES, Task, Ground Truth, Binary Prediction, prediction_prob
"""
from pathlib import Path

import pandas as pd


def load_grover_result(csv_path: Path, task_name: str) -> pd.DataFrame:
    """Read a single GROVER test_result.csv and return a standardised DataFrame."""
    # Skip the second header row (sub-header "ames_mutagenicity") so pandas
    # sees columns: (unnamed), preds, targets
    df = pd.read_csv(csv_path, skiprows=[1])

    # Rename columns to something meaningful
    first_col = df.columns[0]  # unnamed column holding SMILES
    df = df.rename(columns={first_col: "SMILES", "preds": "prediction_prob", "targets": "Ground Truth"})

    df["Task"] = task_name
    df["Ground Truth"] = df["Ground Truth"].astype(int)
    df["Binary Prediction"] = (df["prediction_prob"] >= 0.5).astype(int)

    return df


def load_predictions(results_dir: Path) -> pd.DataFrame:
    """Walk results/<task>/fold_0/test_result.csv and concatenate all tasks."""
    task_dirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {results_dir}")

    dfs = []
    for task_dir in task_dirs:
        csv_path = task_dir / "fold_0" / "test_result.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found – skipping")
            continue
        df = load_grover_result(csv_path, task_name=task_dir.name)
        dfs.append(df)
        print(f"  Loaded {task_dir.name}  ({len(df)} rows)")

    if not dfs:
        raise FileNotFoundError(f"No test_result.csv files found under {results_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    # Keep only the relevant columns in a clean order
    combined = combined[
        ["SMILES", "Task", "Ground Truth", "Binary Prediction", "prediction_prob"]
    ]
    return combined


def main():
    base = Path(__file__).resolve().parent / "GROVER"
    results_dir = base / "results"
    out_dir = base / "predictions_combined"
    out_dir.mkdir(exist_ok=True)

    print(f"Reading STL GROVER predictions from {results_dir}/")
    combined = load_predictions(results_dir)

    out_path = out_dir / "STL_GROVER_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote combined predictions to {out_path.name}  ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
