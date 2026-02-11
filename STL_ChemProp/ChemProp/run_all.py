"""Train all 16 dataset variants and run inference on their test sets."""

import glob
import os
import subprocess
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
TRAIN_DIR = os.path.join(DATA_DIR, "Dataset_Variants_Train_Val")
TEST_DIR = os.path.join(DATA_DIR, "Dataset_Variants_Test")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "predictions")

EPOCHS = 100
BATCH_SIZE = 50
DEVICE = "gpu"


def get_variant_name(train_file: str) -> str:
    """Extract variant name like 'TA100_with_S9' from a training filename."""
    basename = os.path.basename(train_file)
    return basename.replace("_Train_Val.csv", "")


def find_best_checkpoint(variant_dir: str) -> str | None:
    """Find the best checkpoint saved by ModelCheckpoint."""
    pattern = os.path.join(variant_dir, "best", "best-*.ckpt")
    ckpts = sorted(glob.glob(pattern))
    return ckpts[0] if ckpts else None


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"FAILED: {description}")
        return False
    return True


def progress_bar(current, total, width=30):
    """Return a progress bar string."""
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def main():
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*_Train_Val.csv")))
    total = len(train_files)
    print(f"Found {total} training datasets\n")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    failed = []
    start_time = time.time()

    for i, train_file in enumerate(train_files, 1):
        variant = get_variant_name(train_file)
        test_file = os.path.join(TEST_DIR, f"{variant}_Test.csv")

        elapsed = time.time() - start_time
        if i > 1:
            avg_per_variant = elapsed / (i - 1)
            remaining = avg_per_variant * (total - i + 1)
            eta = f"~{remaining/60:.0f}min remaining"
        else:
            eta = ""

        print(f"\n{progress_bar(i - 1, total)}  {eta}")

        if not os.path.exists(test_file):
            print(f"WARNING: No test file for {variant}, skipping")
            failed.append(variant)
            continue

        variant_ckpt_dir = os.path.join(CHECKPOINT_DIR, variant)

        # Train
        success = run_command(
            [
                sys.executable, "train.py",
                "--data", train_file,
                "--epochs", str(EPOCHS),
                "--batch-size", str(BATCH_SIZE),
                "--save-dir", variant_ckpt_dir,
                "--device", DEVICE,
            ],
            f"[{i}/{total}] Training {variant}",
        )
        if not success:
            failed.append(variant)
            continue

        # Find checkpoint
        ckpt = find_best_checkpoint(variant_ckpt_dir)
        if ckpt is None:
            print(f"ERROR: No checkpoint found for {variant}")
            failed.append(variant)
            continue

        # Predict
        output_file = os.path.join(PREDICTIONS_DIR, f"{variant}_predictions.csv")
        success = run_command(
            [
                sys.executable, "predict.py",
                "--data", test_file,
                "--checkpoint", ckpt,
                "--output", output_file,
                "--batch-size", str(BATCH_SIZE),
                "--device", DEVICE,
            ],
            f"[{i}/{total}] Predicting {variant}",
        )
        if not success:
            failed.append(variant)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{progress_bar(total, total)}")
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {total - len(failed)}/{total} succeeded  ({total_time/60:.1f} min)")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print(f"  Predictions: {PREDICTIONS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
