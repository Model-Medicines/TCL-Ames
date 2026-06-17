#!/usr/bin/env python3
"""Run ChemProp inference for all available dataset variants.

This script performs prediction only (no retraining). It matches
`*_Test.csv` files to trained checkpoints and calls `ChemProp/predict.py`.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import Iterable


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Run inference for all STL ChemProp test-set variants.",
    )
    parser.add_argument(
        "--test-dir",
        default=os.path.join(base_dir, "Data", "Dataset_Variants_Test"),
        help="Directory containing files like <variant>_Test.csv",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=os.path.join(base_dir, "ChemProp", "checkpoints"),
        help="Checkpoint root containing variant folders",
    )
    parser.add_argument(
        "--predict-script",
        default=os.path.join(base_dir, "ChemProp", "predict.py"),
        help="Path to existing ChemProp prediction script",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(base_dir, "ChemProp", "predictions"),
        help="Where prediction CSV files are written",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device passed to predict.py",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size passed to predict.py",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional list of variants (e.g., TA97_with_S9 TA100_without_S9)",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero if a test file has no matching checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def variant_from_test_file(test_file: str) -> str | None:
    name = os.path.basename(test_file)
    suffix = "_Test.csv"
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def find_checkpoint(checkpoint_root: str, variant: str) -> str | None:
    pattern = os.path.join(checkpoint_root, variant, "best", "*.ckpt")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def should_run_variant(variant: str, selected_variants: Iterable[str] | None) -> bool:
    if selected_variants is None:
        return True
    return variant in set(selected_variants)


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.predict_script):
        print(f"ERROR: predict script not found: {args.predict_script}")
        return 2
    if not os.path.isdir(args.test_dir):
        print(f"ERROR: test directory not found: {args.test_dir}")
        return 2
    if not os.path.isdir(args.checkpoint_root):
        print(f"ERROR: checkpoint root not found: {args.checkpoint_root}")
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    test_files = sorted(glob.glob(os.path.join(args.test_dir, "*_Test.csv")))
    if not test_files:
        print(f"ERROR: no *_Test.csv files found in {args.test_dir}")
        return 2

    selected = set(args.variants) if args.variants else None
    total = 0
    succeeded = 0
    missing_ckpt: list[str] = []
    failed: list[str] = []

    print(f"Found {len(test_files)} test files in: {args.test_dir}")
    print(f"Checkpoint root: {args.checkpoint_root}")
    print(f"Output directory: {args.output_dir}")

    for test_file in test_files:
        variant = variant_from_test_file(test_file)
        if variant is None:
            continue
        if not should_run_variant(variant, selected):
            continue

        total += 1
        checkpoint = find_checkpoint(args.checkpoint_root, variant)
        if checkpoint is None:
            print(f"SKIP [{variant}] No checkpoint found.")
            missing_ckpt.append(variant)
            continue

        output_file = os.path.join(args.output_dir, f"{variant}_predictions.csv")
        cmd = [
            sys.executable,
            args.predict_script,
            "--data",
            test_file,
            "--checkpoint",
            checkpoint,
            "--output",
            output_file,
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ]

        print("\n" + "=" * 72)
        print(f"Variant: {variant}")
        print(f"Test file: {test_file}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Output: {output_file}")
        print("Command:", " ".join(f'"{token}"' if " " in token else token for token in cmd))

        if args.dry_run:
            succeeded += 1
            continue

        result = subprocess.run(cmd, cwd=os.path.dirname(args.predict_script))
        if result.returncode == 0:
            succeeded += 1
            print(f"OK [{variant}]")
        else:
            failed.append(variant)
            print(f"FAILED [{variant}] exit_code={result.returncode}")

    print("\n" + "=" * 72)
    print(f"Processed variants: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Missing checkpoints: {len(missing_ckpt)}")
    print(f"Failed predictions: {len(failed)}")
    if missing_ckpt:
        print("Missing checkpoint variants:", ", ".join(missing_ckpt))
    if failed:
        print("Failed variants:", ", ".join(failed))

    if args.fail_on_missing and missing_ckpt:
        return 1
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
