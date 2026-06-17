"""Combine per-task foil prediction CSVs into a single summary CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_task_column(df: pd.DataFrame) -> pd.Series:
    return df["Strain"].astype(str) + df["S9"].astype(int).map(
        {1: "_with_S9", 0: "_without_S9"}
    )


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    csv_files = sorted(pred_dir.glob("*_predictions.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_predictions.csv files found in {pred_dir}")

    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["Task"] = build_task_column(df)
        df["Ground Truth"] = pd.NA
        df["Binary Prediction"] = df["prediction"].astype(int)
        dfs.append(df)
        print(f"  Loaded {csv_path.name}  ({len(df)} rows)")

    combined = pd.concat(dfs, ignore_index=True)
    return combined[
        [
            "SMILES",
            "Task",
            "Ground Truth",
            "Binary Prediction",
            "prediction_prob",
        ]
    ].rename(columns={"prediction_prob": "Prediction Probability"})


def main() -> None:
    base = Path(__file__).resolve().parent
    pred_dir = base / "predictions_foil"
    out_dir = base / "predictions_foil_combined"
    out_dir.mkdir(exist_ok=True)

    print(f"Reading foil predictions from {pred_dir}/")
    combined = load_predictions(pred_dir)

    out_path = out_dir / "STL_RF_foil_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote combined foil predictions to {out_path.name}  ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
