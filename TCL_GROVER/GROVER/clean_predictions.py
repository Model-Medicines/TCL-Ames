"""
Clean GROVER test results by joining predictions from
results/fold_0/test_result.csv with metadata from
Data/test_ood_fixed_df.csv (aligned row-by-row).

Columns produced:
    gmtamesQSAR_ID    – compound ID
    SMILES            – molecular SMILES string
    Task              – Strain + S9 status, e.g. 'TA100_without_S9'
    Ground Truth      – Endpoint mapped to binary (Positive=1, Negative=0)
    Binary Prediction – prediction probability thresholded at 0.5
    prediction_prob   – raw prediction probability
"""
from pathlib import Path

import pandas as pd


def main():
    base = Path(__file__).resolve().parent
    pred_path = base / "results" / "fold_0" / "test_result.csv"
    meta_path = base.parent / "Data" / "test_ood_fixed_df.csv"
    out_dir = base / "cleaned_predictions"
    out_dir.mkdir(exist_ok=True)

    # Read predictions, skipping the task-name sub-header row
    print(f"Reading predictions from {pred_path}")
    preds = pd.read_csv(pred_path, skiprows=[1])
    smiles_col = preds.columns[0]
    preds = preds.rename(columns={smiles_col: "SMILES"})
    print(f"  Loaded {len(preds)} prediction rows")

    # Read test metadata for Task info
    print(f"Reading metadata from {meta_path}")
    meta = pd.read_csv(meta_path)
    print(f"  Loaded {len(meta)} metadata rows")

    assert len(preds) == len(meta), (
        f"Row count mismatch: {len(preds)} predictions vs {len(meta)} metadata rows"
    )

    # Build output dataframe using metadata + predictions (aligned by row)
    df = pd.DataFrame()
    df["gmtamesQSAR_ID"] = meta["gmtamesQSAR_ID"].values
    df["SMILES"] = meta["SMILES"].values
    df["Task"] = (meta["Strain"] + meta["S9"].map({1: "_with_S9", 0: "_without_S9"})).values
    df["Ground Truth"] = meta["Endpoint"].map({"Positive": 1, "Negative": 0}).values
    df["Binary Prediction"] = (preds["preds"] >= 0.5).astype(int).values
    df["prediction_prob"] = preds["preds"].values

    out_path = out_dir / "GROVER_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.name}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
