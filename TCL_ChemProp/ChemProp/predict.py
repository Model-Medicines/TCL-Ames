"""Task-conditioned ChemProp v2 prediction script."""

import argparse

import lightning as L
import numpy as np
import pandas as pd
from rdkit import Chem

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN

STRAINS = ["TA1535", "TA1538", "TA104", "TA97", "TA98", "TA100", "TA102", "TA1537"]


def ames_condition_vector(strain: str, s9: int) -> np.ndarray:
    """Build a 9-dim condition vector: 8 one-hot strain + 1 binary S9."""
    vec = np.zeros(9, dtype=np.float32)
    vec[STRAINS.index(strain)] = 1.0
    vec[8] = float(s9)
    return vec


def main():
    parser = argparse.ArgumentParser(description="Run predictions with a trained ChemProp model")
    parser.add_argument("--data", required=True, help="Path to test CSV with SMILES column")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"], help="Device")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = MPNN.load_from_checkpoint(args.checkpoint)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    datapoints = []
    valid_indices = []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        x_d = ames_condition_vector(row["Strain"], row["S9"])
        datapoints.append(MoleculeDatapoint(mol=mol, x_d=x_d))
        valid_indices.append(i)

    dset = MoleculeDataset(datapoints)
    loader = build_dataloader(dset, batch_size=args.batch_size, shuffle=False)

    # Predict using Lightning Trainer
    print("Running predictions...")
    trainer = L.Trainer(accelerator=args.device, logger=False, enable_progress_bar=True)
    batch_preds = trainer.predict(model, loader)
    preds = np.concatenate([p.cpu().numpy() for p in batch_preds], axis=0)

    # Save results
    results = df.loc[valid_indices].copy().reset_index(drop=True)
    results["prediction_prob"] = preds[:, 0]
    results["prediction"] = (preds[:, 0] >= 0.5).astype(int)
    results["prediction_label"] = results["prediction"].map({0: "Negative", 1: "Positive"})
    results.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
