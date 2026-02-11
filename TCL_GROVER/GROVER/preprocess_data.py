"""
Preprocess combined Ames data for Task-Conditioned GROVER fine-tuning.

Produces:
  1. GROVER-compatible CSVs (smiles, ames_mutagenicity)
  2. .npz feature files with 9-dim condition vectors (8 strain one-hot + 1 S9)
     aligned row-by-row with the CSVs

The condition vectors are passed to GROVER via --features_path and get
concatenated with the encoder output before the FFN heads.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
OUT_DIR = os.path.join(BASE_DIR, "data_processed")

STRAINS = ["TA1535", "TA1538", "TA104", "TA97", "TA98", "TA100", "TA102", "TA1537"]
ENDPOINT_MAP = {"Positive": 1, "Negative": 0}


def ames_condition_vector(strain: str, s9: int) -> np.ndarray:
    """Build a 9-dim condition vector: 8 one-hot strain + 1 binary S9."""
    vec = np.zeros(9, dtype=np.float32)
    vec[STRAINS.index(strain)] = 1.0
    vec[8] = float(s9)
    return vec


def process_split(csv_path, out_csv_name, out_npz_name):
    """Convert a combined CSV into GROVER CSV + condition features .npz."""
    df = pd.read_csv(csv_path)

    # Build GROVER-compatible CSV
    grover_df = pd.DataFrame({
        "smiles": df["SMILES"],
        "ames_mutagenicity": df["Endpoint"].map(ENDPOINT_MAP),
    })

    # Build condition feature matrix (N x 9)
    features = np.array(
        [ames_condition_vector(row["Strain"], row["S9"]) for _, row in df.iterrows()],
        dtype=np.float32,
    )

    # Write outputs
    csv_out = os.path.join(OUT_DIR, out_csv_name)
    npz_out = os.path.join(OUT_DIR, out_npz_name)

    grover_df.to_csv(csv_out, index=False)
    np.savez(npz_out, features=features)

    # Summary
    n_pos = (grover_df["ames_mutagenicity"] == 1).sum()
    n_neg = (grover_df["ames_mutagenicity"] == 0).sum()
    strains_present = df["Strain"].nunique()
    print(f"  {out_csv_name}: {len(df)} samples ({n_pos} pos, {n_neg} neg), "
          f"{strains_present} strains, features shape={features.shape}")
    print(f"  -> {csv_out}")
    print(f"  -> {npz_out}")

    return features


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Processing train/val data...")
    train_features = process_split(
        os.path.join(DATA_DIR, "train_val_ood_fixed_df.csv"),
        "train_val.csv",
        "train_val_features.npz",
    )

    print("\nProcessing test data...")
    test_features = process_split(
        os.path.join(DATA_DIR, "test_ood_fixed_df.csv"),
        "test.csv",
        "test_features.npz",
    )

    # Verify condition vectors
    print("\nCondition vector verification:")
    print(f"  Train feature range: [{train_features.min()}, {train_features.max()}]")
    print(f"  Test  feature range: [{test_features.min()}, {test_features.max()}]")
    print(f"  Unique train vectors: {len(np.unique(train_features, axis=0))}")
    print(f"  Unique test  vectors: {len(np.unique(test_features, axis=0))}")
    print(f"\n  Strain order: {STRAINS}")
    print(f"  Vector format: [strain_one_hot (8-dim), S9 (1-dim)]")

    print("\nDone.")
