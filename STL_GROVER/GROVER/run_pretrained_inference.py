"""
Run GROVER inference for each pretrained STL model on matching test CSVs.

Expected defaults:
  - Models: pretrained_models/*_model.pt
  - Test data: data_processed/test/*.csv
  - Outputs: predictions_pretrained/*_predictions.csv
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "pretrained_models"
DEFAULT_TEST_DIR = BASE_DIR / "data_processed" / "test"
DEFAULT_OUTPUT_DIR = BASE_DIR / "predictions_pretrained"
DEFAULT_MAIN = BASE_DIR / "main.py"


def model_key(model_path: Path) -> str:
    """Map model filename to variant key."""
    stem = model_path.stem
    return stem[:-6] if stem.endswith("_model") else stem


def discover_models(model_dir: Path) -> dict[str, Path]:
    models = {}
    for path in sorted(model_dir.glob("*.pt")):
        models[model_key(path)] = path
    return models


def discover_test_files(test_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(test_dir.glob("*.csv"))}


@contextmanager
def single_model_checkpoint_dir(checkpoint_path: Path):
    """
    Create a temporary checkpoint directory containing one model file.

    This script shell-calls ``main.py predict`` which expects ``--checkpoint_dir``
    in this codebase. To run one specific model per variant, we create a temporary
    directory with just that model and pass it as the checkpoint directory.
    """
    with tempfile.TemporaryDirectory(prefix="grover_single_ckpt_") as tmp:
        tmp_dir = Path(tmp)
        staged_model_path = tmp_dir / checkpoint_path.name
        try:
            staged_model_path.symlink_to(checkpoint_path)
        except OSError:
            # Fall back to copy if symlinks are unavailable.
            shutil.copy2(checkpoint_path, staged_model_path)
        yield tmp_dir


def run_prediction(
    python_exec: str,
    main_script: Path,
    checkpoint_path: Path,
    data_path: Path,
    output_path: Path,
    gpu: int,
) -> None:
    with single_model_checkpoint_dir(checkpoint_path) as checkpoint_dir:
        cmd = [
            python_exec,
            str(main_script),
            "predict",
            "--data_path",
            str(data_path),
            "--output_path",
            str(output_path),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--no_features_scaling",
            "--gpu",
            str(gpu),
        ]
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained GROVER checkpoints on matching test CSVs."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing .pt model files (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help=f"Directory containing test CSV files (default: {DEFAULT_TEST_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write prediction CSVs (default: {DEFAULT_OUTPUT_DIR}).",
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

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
    if not args.main_script.exists():
        raise FileNotFoundError(f"main.py not found: {args.main_script}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(args.model_dir)
    test_files = discover_test_files(args.test_dir)

    if not models:
        raise FileNotFoundError(f"No .pt models found in {args.model_dir}")
    if not test_files:
        raise FileNotFoundError(f"No .csv files found in {args.test_dir}")

    common_variants = sorted(set(models) & set(test_files))
    missing_models = sorted(set(test_files) - set(models))
    missing_tests = sorted(set(models) - set(test_files))

    print(f"Found {len(models)} model(s), {len(test_files)} test file(s).")
    print(f"Running paired inference for {len(common_variants)} variant(s).")

    if missing_models:
        print(f"Warning: missing models for variants: {', '.join(missing_models)}")
    if missing_tests:
        print(f"Warning: missing test CSVs for variants: {', '.join(missing_tests)}")
    if not common_variants:
        raise RuntimeError("No overlapping model/test variant names were found.")

    for idx, variant in enumerate(common_variants, start=1):
        model_path = models[variant]
        test_path = test_files[variant]
        output_path = args.output_dir / f"{variant}_predictions.csv"

        print(f"[{idx}/{len(common_variants)}] {variant}")
        print(f"  model: {model_path.name}")
        print(f"  data:  {test_path.name}")
        print(f"  out:   {output_path.name}")

        run_prediction(
            python_exec=args.python_exec,
            main_script=args.main_script,
            checkpoint_path=model_path,
            data_path=test_path,
            output_path=output_path,
            gpu=args.gpu,
        )

    print("\nDone.")
    print(f"Predictions written to: {args.output_dir}")


if __name__ == "__main__":
    main()
