#!/usr/bin/env python3
"""Train a single-task random forest on ECFP4 fingerprints."""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import build_feature_matrix, is_valid_smiles


def load_labeled_data(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    valid_mask = df["SMILES"].astype(str).apply(is_valid_smiles)
    df = df[valid_mask].reset_index(drop=True)
    y = (df["Endpoint"].astype(str) == "Positive").astype(int).to_numpy()
    return df, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-task random forest")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument(
        "--save-dir",
        default="models",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees in the forest",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df, y = load_labeled_data(args.data)
    print(f"Loaded {len(df)} valid molecules")

    x = build_feature_matrix(df)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)[:, 1]
    if len(np.unique(y_val)) > 1:
        val_auc = roc_auc_score(y_val, val_probs)
        print(f"Validation ROC-AUC: {val_auc:.4f}")
    else:
        print("Validation ROC-AUC: n/a (single class in validation split)")

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
