#!/usr/bin/env python
"""
Create all_features.csv from featurized Train/Test CSV files.

This script scans both:
  - Train_Data_Featurized
  - Test_Data_Featurized

and writes a one-column CSV named all_features.csv containing descriptor
feature names (for example D001..D777).
"""

import argparse
import glob
import os
import pandas as pd


NON_FEATURE_COLUMNS = {"gmtamesQSAR_ID", "SMILES", "label"}


def extract_feature_columns(csv_path):
    """
    Read only the header from one CSV and return descriptor column names.
    """
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return [col for col in columns if col not in NON_FEATURE_COLUMNS]


def collect_feature_files(train_dir, test_dir):
    """
    Collect all CSV files from train and test featurized directories.
    """
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    return train_files + test_files


def build_feature_list(csv_files):
    """
    Build one ordered feature list across all input CSV headers.

    Preserves first-seen order and avoids duplicates.
    """
    ordered_features = []
    seen = set()

    for csv_path in csv_files:
        features = extract_feature_columns(csv_path)
        for feature in features:
            if feature not in seen:
                seen.add(feature)
                ordered_features.append(feature)

    return ordered_features


def main():
    parser = argparse.ArgumentParser(
        description="Create all_features.csv from featurized train/test files."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help=(
            "Directory that contains Train_Data_Featurized and "
            "Test_Data_Featurized. Default: script directory."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional explicit output CSV path. Default: <base-dir>/all_features.csv",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.base_dir, "Train_Data_Featurized")
    test_dir = os.path.join(args.base_dir, "Test_Data_Featurized")
    output_file = (
        args.output_file
        if args.output_file
        else os.path.join(args.base_dir, "all_features.csv")
    )

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    csv_files = collect_feature_files(train_dir, test_dir)
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in Train_Data_Featurized/Test_Data_Featurized."
        )

    print(f"Found {len(csv_files)} featurized files")
    print(f"Train dir: {train_dir}")
    print(f"Test dir:  {test_dir}")

    features = build_feature_list(csv_files)
    if not features:
        raise ValueError("No feature columns found in provided CSV files.")

    pd.DataFrame({"feature": features}).to_csv(output_file, index=False)

    print(f"Saved feature list to: {output_file}")
    print(f"Total features: {len(features)}")
    if len(features) >= 3:
        print(f"First 3: {features[:3]}")
        print(f"Last 3:  {features[-3:]}")


if __name__ == "__main__":
    main()
