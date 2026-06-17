#!/usr/bin/env python3
"""Score test datasets with the trained random forest model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import build_feature_matrix

MODEL_PATH = Path("/Users/tylerumansky/Desktop/RF baseline/models/rf_ecfp_strain_s9.joblib")
TEST1_PATH = Path("/Users/tylerumansky/Desktop/RF baseline/Data/test_ood_master_df.csv")
TEST2_PATH = Path("/Users/tylerumansky/Desktop/RF baseline/Data/foil_long.csv")
OUTPUT1_PATH = Path("/Users/tylerumansky/Desktop/RF baseline/outputs/test_ood_master_df_predictions.csv")
OUTPUT2_PATH = Path("/Users/tylerumansky/Desktop/RF baseline/outputs/foil_long_predictions.csv")


def make_task_column(df: pd.DataFrame) -> pd.Series:
    s9_label = df["S9"].astype(int).map({0: "without_S9", 1: "with_S9"})
    return df["Strain"].astype(str) + "_" + s9_label


def format_predictions(df: pd.DataFrame, prediction_prob: pd.Series) -> pd.DataFrame:
    output_df = pd.DataFrame(
        {
            "gmtamesQSAR_ID": df["gmtamesQSAR_ID"] if "gmtamesQSAR_ID" in df.columns else "",
            "SMILES": df["SMILES"],
            "Task": make_task_column(df),
            "Ground Truth": (
                (df["Endpoint"].astype(str) == "Positive").astype(int)
                if "Endpoint" in df.columns
                else pd.NA
            ),
            "Binary Prediction": (prediction_prob >= 0.5).astype(int),
            "prediction_prob": prediction_prob,
        }
    )
    if "gmtamesQSAR_ID" not in df.columns:
        output_df["gmtamesQSAR_ID"] = ""
    return output_df[
        [
            "gmtamesQSAR_ID",
            "SMILES",
            "Task",
            "Ground Truth",
            "Binary Prediction",
            "prediction_prob",
        ]
    ]


def score_dataset(
    input_path: Path,
    output_path: Path,
    model_bundle: dict,
) -> None:
    df = pd.read_csv(input_path)
    x_test = build_feature_matrix(
        df,
        model_bundle["strain_categories"],
        model_bundle["s9_categories"],
    )
    prediction_prob = pd.Series(
        model_bundle["model"].predict_proba(x_test)[:, 1],
        index=df.index,
    )
    output_df = format_predictions(df, prediction_prob)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {len(output_df)} predictions to {output_path}")


def main() -> None:
    model_bundle = joblib.load(MODEL_PATH)

    score_dataset(TEST1_PATH, OUTPUT1_PATH, model_bundle)
    score_dataset(TEST2_PATH, OUTPUT2_PATH, model_bundle)


if __name__ == "__main__":
    main()
