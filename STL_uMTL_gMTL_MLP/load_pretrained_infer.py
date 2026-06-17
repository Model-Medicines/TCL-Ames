"""
Load pretrained gmtames models and run inference on the test split.

This script rebuilds the network architecture from each saved
`*_hyperparam_dict.json`, loads the corresponding `*_state_dict.pt`,
and saves test predictions to pickle files compatible with `gmtames results`.

Example:
    python load_pretrained_infer.py --testsplit scaffold --device cpu

Run from project root:
    /home/ubuntu/Desktop/ACS CRT Models (updated)/STL_uMTL_gMTL_MLP
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from gmtames.data import generateModellingDatasets
from gmtames.nn import NTaskNeuralNetworkFromDict, checkDevice, loadModellingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-set inference using pretrained gmtames models."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("output/mtg_experiment_scaffold/final_models"),
        help="Directory containing *_state_dict.pt and *_hyperparam_dict.json files.",
    )
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=None,
        help="Optional specific *_state_dict.pt path. If omitted, all models in --models-dir are used.",
    )
    parser.add_argument(
        "--testsplit",
        default="scaffold",
        help='Dataset split name used in base datasets (default: "scaffold").',
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device (e.g., "cpu", "cuda:0", "mps"). Defaults to "cpu".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/mtg_experiment_scaffold/test_predictions"),
        help="Directory to write *_test_predictions.pkl files.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a wide CSV per model with y_true/y_pred columns.",
    )
    return parser.parse_args()


def model_id_from_state_dict(path: Path) -> str:
    suffix = "_state_dict.pt"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected file ending with {suffix}: {path}")
    return path.name[: -len(suffix)]


def run_inference_for_state_dict(
    state_dict_path: Path,
    testsplit: str,
    device: str,
    output_dir: Path,
    write_csv: bool,
) -> Path:
    model_id = model_id_from_state_dict(state_dict_path)
    hyperparam_path = state_dict_path.with_name(f"{model_id}_hyperparam_dict.json")
    if not hyperparam_path.exists():
        raise FileNotFoundError(f"Missing hyperparameter file: {hyperparam_path}")

    with hyperparam_path.open("r", encoding="utf-8") as f:
        hyperparams = json.load(f)

    tasks = hyperparams.get("tasks")
    if not tasks:
        raise ValueError(f"`tasks` missing in hyperparameter file: {hyperparam_path}")

    modelling_datasets, _ = generateModellingDatasets(",".join(tasks), testsplit)
    gmtames_ids, test_loader = loadModellingDataset(modelling_datasets, tasks, "test", device)

    x_test, y_true_t = next(iter(test_loader))
    n_input = x_test.size(dim=1)

    model = NTaskNeuralNetworkFromDict(hyperparams, n_input, tasks).to(device)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        y_pred_t = model(x_test)

    y_true = y_true_t.detach().cpu().numpy()
    y_pred = y_pred_t.detach().cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / f"{model_id}_test_predictions.pkl"
    payload = {
        "gmtamesqsar_id": np.asarray(gmtames_ids),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)

    if write_csv:
        csv_path = output_dir / f"{model_id}_test_predictions.csv"
        df = pd.DataFrame({"gmtamesQSAR_ID": np.asarray(gmtames_ids)})
        for i, task in enumerate(tasks):
            df[f"{task}_y_true"] = y_true[:, i]
            df[f"{task}_y_pred"] = y_pred[:, i]
        df.to_csv(csv_path, index=False)

    return pkl_path


def discover_state_dicts(models_dir: Path, single_state_dict: Optional[Path]) -> List[Path]:
    if single_state_dict is not None:
        return [single_state_dict]
    return sorted(models_dir.glob("*_state_dict.pt"))


def main() -> None:
    args = parse_args()

    # Normalize and validate device against project's existing utility.
    device = checkDevice(args.device)

    state_dict_paths = discover_state_dicts(args.models_dir, args.state_dict)
    if not state_dict_paths:
        raise FileNotFoundError(
            f"No model checkpoints found. Looked in: {args.models_dir.resolve()}"
        )

    print(f"Using device: {device}")
    print(f"Found {len(state_dict_paths)} model checkpoint(s).")

    for state_dict_path in state_dict_paths:
        out_path = run_inference_for_state_dict(
            state_dict_path=state_dict_path,
            testsplit=args.testsplit,
            device=device,
            output_dir=args.output_dir,
            write_csv=args.write_csv,
        )
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
