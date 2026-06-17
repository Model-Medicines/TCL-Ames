"""
Run single-pass GROVER inference using one pretrained checkpoint directory.

Expected defaults:
  - Checkpoints: pretrained_models/
  - Test data: data_processed/test.csv
  - Test features: data_processed/test_features.npz
  - Output: predictions_pretrained/TCL_GROVER_predictions.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "pretrained_models"
DEFAULT_DATA_PATH = BASE_DIR / "data_processed" / "test.csv"
DEFAULT_FEATURES_PATH = BASE_DIR / "data_processed" / "test_features.npz"
DEFAULT_LABELS_PATH = BASE_DIR.parent / "Data" / "test_ood_master_df.csv"
DEFAULT_OUTPUT_PATH = (
    BASE_DIR / "predictions_pretrained" / "TCL_GROVER_predictions.csv"
)
DEFAULT_MAIN = BASE_DIR / "main.py"


def run_prediction(
    python_exec: str,
    main_script: Path,
    checkpoint_dir: Path,
    data_path: Path,
    features_path: Path,
    output_path: Path,
    gpu: int,
) -> None:
    cmd = [
        python_exec,
        str(main_script),
        "predict",
        "--data_path",
        str(data_path),
        "--features_path",
        str(features_path),
        "--output_path",
        str(output_path),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--no_features_scaling",
        "--gpu",
        str(gpu),
    ]
    subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


def _task_name(strain: str, s9: int) -> str:
    return f"{strain}_{'with_S9' if int(s9) == 1 else 'without_S9'}"


def reformat_output(output_path: Path, labels_path: Path) -> None:
    pred_df = pd.read_csv(output_path)
    if pred_df.shape[1] < 2:
        raise ValueError(f"Unexpected prediction format in {output_path}")

    first_col = pred_df.columns[0]  # unnamed SMILES index column
    score_col = "ames_mutagenicity" if "ames_mutagenicity" in pred_df.columns else pred_df.columns[1]
    pred_df = pred_df.rename(columns={first_col: "SMILES", score_col: "prediction_prob"})

    labels_df = pd.read_csv(labels_path)
    required_cols = {"SMILES", "Strain", "S9", "Endpoint"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in labels file: {missing_cols}")

    labels_df = labels_df[["SMILES", "Strain", "S9", "Endpoint"]].copy()
    labels_df["Task"] = [
        _task_name(strain, s9) for strain, s9 in zip(labels_df["Strain"], labels_df["S9"])
    ]
    labels_df["Ground Truth"] = labels_df["Endpoint"].map({"Positive": 1, "Negative": 0})
    if labels_df["Ground Truth"].isna().any():
        raise ValueError("Endpoint contains values outside {'Positive', 'Negative'}.")

    if len(pred_df) != len(labels_df):
        raise ValueError(
            "Prediction and labels row counts do not match. "
            f"pred={len(pred_df)}, labels={len(labels_df)}"
        )
    if not pred_df["SMILES"].equals(labels_df["SMILES"]):
        raise ValueError(
            "SMILES order mismatch between prediction output and labels file. "
            "Cannot safely align Task/Ground Truth."
        )

    formatted_df = pd.DataFrame(
        {
            "SMILES": pred_df["SMILES"],
            "Task": labels_df["Task"].astype(str).values,
            "Ground Truth": labels_df["Ground Truth"].astype(int).values,
            "Binary Prediction": (pred_df["prediction_prob"].astype(float) >= 0.5).astype(int),
            "prediction_prob": pred_df["prediction_prob"].astype(float),
        }
    )
    formatted_df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single pretrained GROVER inference."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=(
            "Directory containing pretrained .pt checkpoint file(s) "
            f"(default: {DEFAULT_CHECKPOINT_DIR})."
        ),
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Input CSV for inference (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--features_path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help=f"Input .npz features for inference (default: {DEFAULT_FEATURES_PATH}).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--labels_path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"Labels CSV used to format output columns (default: {DEFAULT_LABELS_PATH}).",
    )
    parser.add_argument(
        "--main_script",
        type=Path,
        default=DEFAULT_MAIN,
        help=f"Path to GROVER main.py (default: {DEFAULT_MAIN}).",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to launch main.py predict.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index passed to GROVER predict (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.checkpoint_dir}"
        )
    if not args.data_path.exists():
        raise FileNotFoundError(f"Input data CSV not found: {args.data_path}")
    if not args.features_path.exists():
        raise FileNotFoundError(f"Input features .npz not found: {args.features_path}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {args.labels_path}")
    if not args.main_script.exists():
        raise FileNotFoundError(f"main.py not found: {args.main_script}")

    checkpoint_files = sorted(args.checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pt checkpoints found in {args.checkpoint_dir}")

    if len(checkpoint_files) > 1:
        print(
            f"Warning: found {len(checkpoint_files)} checkpoint files; "
            "GROVER will use all checkpoints present in the directory."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running single inference...")
    print(f"  checkpoint_dir: {args.checkpoint_dir}")
    print(f"  data_path:      {args.data_path}")
    print(f"  features_path:  {args.features_path}")
    print(f"  labels_path:    {args.labels_path}")
    print(f"  output_path:    {args.output_path}")

    run_prediction(
        python_exec=args.python_exec,
        main_script=args.main_script,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        features_path=args.features_path,
        output_path=args.output_path,
        gpu=args.gpu,
    )
    reformat_output(args.output_path, args.labels_path)

    print("\nDone.")
    print(f"Predictions written to: {args.output_path}")


if __name__ == "__main__":
    main()
