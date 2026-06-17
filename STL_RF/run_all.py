#!/usr/bin/env python3
"""Train all 16 dataset variants and run inference on their test and foil sets."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
TRAIN_DIR = os.path.join(DATA_DIR, "Dataset_Variants_Train_Val")
TEST_DIR = os.path.join(DATA_DIR, "Dataset_Variants_Test")
FOIL_DIR = os.path.join(DATA_DIR, "Foil_Dataset_Variants_Test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
FOIL_PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions_foil")

N_ESTIMATORS = 500


def get_variant_name(train_file: str) -> str:
    basename = os.path.basename(train_file)
    return basename.replace("_Train_Val.csv", "")


def find_model_path(variant_dir: str) -> str | None:
    model_path = os.path.join(variant_dir, "model.joblib")
    return model_path if os.path.exists(model_path) else None


def run_command(cmd: list[str], description: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"FAILED: {description}")
        return False
    return True


def progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def main() -> None:
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*_Train_Val.csv")))
    total = len(train_files)
    print(f"Found {total} training datasets\n")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(FOIL_PREDICTIONS_DIR, exist_ok=True)

    failed: list[str] = []
    start_time = time.time()

    for i, train_file in enumerate(train_files, 1):
        variant = get_variant_name(train_file)
        test_file = os.path.join(TEST_DIR, f"{variant}_Test.csv")
        foil_file = os.path.join(FOIL_DIR, f"{variant}_Test.csv")

        elapsed = time.time() - start_time
        if i > 1:
            avg_per_variant = elapsed / (i - 1)
            remaining = avg_per_variant * (total - i + 1)
            eta = f"~{remaining / 60:.0f}min remaining"
        else:
            eta = ""

        print(f"\n{progress_bar(i - 1, total)}  {eta}")

        if not os.path.exists(test_file):
            print(f"WARNING: No test file for {variant}, skipping")
            failed.append(variant)
            continue

        variant_model_dir = os.path.join(MODEL_DIR, variant)

        success = run_command(
            [
                sys.executable,
                "train.py",
                "--data",
                train_file,
                "--save-dir",
                variant_model_dir,
                "--n-estimators",
                str(N_ESTIMATORS),
            ],
            f"[{i}/{total}] Training {variant}",
        )
        if not success:
            failed.append(variant)
            continue

        model_path = find_model_path(variant_model_dir)
        if model_path is None:
            print(f"ERROR: No model found for {variant}")
            failed.append(variant)
            continue

        output_file = os.path.join(PREDICTIONS_DIR, f"{variant}_predictions.csv")
        success = run_command(
            [
                sys.executable,
                "predict.py",
                "--data",
                test_file,
                "--model",
                model_path,
                "--output",
                output_file,
            ],
            f"[{i}/{total}] Predicting {variant} (test)",
        )
        if not success:
            failed.append(variant)
            continue

        if os.path.exists(foil_file):
            foil_output = os.path.join(FOIL_PREDICTIONS_DIR, f"{variant}_predictions.csv")
            success = run_command(
                [
                    sys.executable,
                    "predict.py",
                    "--data",
                    foil_file,
                    "--model",
                    model_path,
                    "--output",
                    foil_output,
                ],
                f"[{i}/{total}] Predicting {variant} (foil)",
            )
            if not success:
                failed.append(f"{variant}_foil")

    total_time = time.time() - start_time
    print(f"\n{progress_bar(total, total)}")
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {total - len(failed)}/{total} succeeded  ({total_time / 60:.1f} min)")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"  Models: {MODEL_DIR}/")
    print(f"  Predictions: {PREDICTIONS_DIR}/")
    print(f"  Foil predictions: {FOIL_PREDICTIONS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
