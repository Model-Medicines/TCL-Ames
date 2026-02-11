"""Train task-conditioned ChemProp and run inference on the test set."""

import glob
import os
import subprocess
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
TRAIN_FILE = os.path.join(DATA_DIR, "train_val_ood_fixed_df.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_ood_fixed_df.csv")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "predictions")

EPOCHS = 100
BATCH_SIZE = 50
DEVICE = "gpu"


def find_best_checkpoint(ckpt_dir: str) -> str | None:
    """Find the best checkpoint saved by ModelCheckpoint."""
    pattern = os.path.join(ckpt_dir, "best", "best-*.ckpt")
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


def main():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    start_time = time.time()

    # Train
    success = run_command(
        [
            sys.executable, "train.py",
            "--data", TRAIN_FILE,
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--save-dir", CHECKPOINT_DIR,
            "--device", DEVICE,
        ],
        "Training task-conditioned model",
    )
    if not success:
        print("Training failed.")
        return

    # Find checkpoint
    ckpt = find_best_checkpoint(CHECKPOINT_DIR)
    if ckpt is None:
        print("ERROR: No checkpoint found")
        return

    # Predict
    output_file = os.path.join(PREDICTIONS_DIR, "test_predictions.csv")
    success = run_command(
        [
            sys.executable, "predict.py",
            "--data", TEST_FILE,
            "--checkpoint", ckpt,
            "--output", output_file,
            "--batch-size", str(BATCH_SIZE),
            "--device", DEVICE,
        ],
        "Running inference on test set",
    )

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    if success:
        print(f"  COMPLETE ({total_time/60:.1f} min)")
        print(f"  Checkpoint: {ckpt}")
        print(f"  Predictions: {output_file}")
    else:
        print(f"  Inference failed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
