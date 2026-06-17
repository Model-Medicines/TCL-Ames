#!/usr/bin/env python3
"""Run inference with a trained single-task random forest."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_feature_matrix, is_valid_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions with a trained random forest")
    parser.add_argument("--data", required=True, help="Path to test CSV with SMILES column")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    valid_mask = df["SMILES"].astype(str).apply(is_valid_smiles)
    valid_indices = df.index[valid_mask].tolist()
    valid_df = df.loc[valid_indices].reset_index(drop=True)

    print(f"Running predictions on {len(valid_df)} valid molecules...")
    x = build_feature_matrix(valid_df)
    probs = model.predict_proba(x)[:, 1]

    results = valid_df.copy()
    results["prediction_prob"] = probs
    results["prediction"] = (probs >= 0.5).astype(int)
    results["prediction_label"] = results["prediction"].map({0: "Negative", 1: "Positive"})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
