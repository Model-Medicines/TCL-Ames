"""
Preprocess STL Ames data into GROVER-compatible CSV format.

Input:  CSVs with columns [gmtamesQSAR_ID, SMILES, Strain, S9, Endpoint]
Output: CSVs with columns [smiles, ames_mutagenicity]  (Positive->1, Negative->0)
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TRAIN_DIR = os.path.join(BASE_DIR, "..", "Data", "Dataset_Variants_Train_Val")
INPUT_TEST_DIR = os.path.join(BASE_DIR, "..", "Data", "Dataset_Variants_Test")
OUTPUT_TRAIN_DIR = os.path.join(BASE_DIR, "data_processed", "train_val")
OUTPUT_TEST_DIR = os.path.join(BASE_DIR, "data_processed", "test")

ENDPOINT_MAP = {"Positive": 1, "Negative": 0}


def process_directory(input_dir, output_dir, suffix_to_strip):
    os.makedirs(output_dir, exist_ok=True)
    processed = 0

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(input_dir, fname))
        out_df = pd.DataFrame({
            "smiles": df["SMILES"],
            "ames_mutagenicity": df["Endpoint"].map(ENDPOINT_MAP)
        })

        # Strip the suffix (e.g., "_Train_Val" or "_Test") from filename
        variant_name = fname.replace(suffix_to_strip, ".csv")
        out_path = os.path.join(output_dir, variant_name)
        out_df.to_csv(out_path, index=False)

        n_pos = (out_df["ames_mutagenicity"] == 1).sum()
        n_neg = (out_df["ames_mutagenicity"] == 0).sum()
        print(f"  {variant_name}: {len(out_df)} samples ({n_pos} pos, {n_neg} neg, "
              f"{100*n_pos/len(out_df):.1f}% positive)")
        processed += 1

    return processed


if __name__ == "__main__":
    print("Processing train/val data...")
    n = process_directory(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR, "_Train_Val.csv")
    print(f"  -> {n} files written to {OUTPUT_TRAIN_DIR}\n")

    print("Processing test data...")
    n = process_directory(INPUT_TEST_DIR, OUTPUT_TEST_DIR, "_Test.csv")
    print(f"  -> {n} files written to {OUTPUT_TEST_DIR}\n")

    print("Done.")
