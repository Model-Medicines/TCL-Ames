#!/usr/bin/env python3
"""Train a random forest on ECFP + Strain/S9 one-hot features."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import build_feature_matrix

TRAIN_PATH = Path("/Users/tylerumansky/Desktop/RF baseline/Data/train_val_ood_master_df.csv")
MODEL_PATH = Path("/Users/tylerumansky/Desktop/RF baseline/models/rf_ecfp_strain_s9.joblib")


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)

    strain_categories = sorted(train_df["Strain"].astype(str).unique())
    s9_categories = sorted(train_df["S9"].astype(int).unique())

    x_train = build_feature_matrix(train_df, strain_categories, s9_categories)
    y_train = (train_df["Endpoint"].astype(str) == "Positive").astype(int).to_numpy()

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "strain_categories": strain_categories,
            "s9_categories": s9_categories,
        },
        MODEL_PATH,
    )
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
