"""Task-conditioned ChemProp v2 training script for binary classification.

Encodes Strain (8 one-hot) + S9 (1 binary) as a 9-dim condition vector
that is concatenated with the molecular encoding before the FFN.
"""

import argparse
import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from rdkit import Chem

from sklearn.model_selection import train_test_split

from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    build_dataloader,
)
from chemprop.models import MPNN
from chemprop.nn import (
    BinaryClassificationFFN,
    BondMessagePassing,
    NormAggregation,
)


STRAINS = ["TA1535", "TA1538", "TA104", "TA97", "TA98", "TA100", "TA102", "TA1537"]


def ames_condition_vector(strain: str, s9: int) -> np.ndarray:
    """Build a 9-dim condition vector: 8 one-hot strain + 1 binary S9."""
    vec = np.zeros(9, dtype=np.float32)
    vec[STRAINS.index(strain)] = 1.0
    vec[8] = float(s9)
    return vec


def load_data(csv_path: str) -> list[MoleculeDatapoint]:
    """Load a CSV with SMILES, Strain, S9, and Endpoint columns."""
    df = pd.read_csv(csv_path)
    label_map = {"Negative": 0, "Positive": 1}
    datapoints = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        y = np.array([label_map[row["Endpoint"]]], dtype=np.float32)
        x_d = ames_condition_vector(row["Strain"], row["S9"])
        datapoints.append(MoleculeDatapoint(mol=mol, y=y, x_d=x_d))
    return datapoints


def main():
    parser = argparse.ArgumentParser(description="Train a ChemProp v2 binary classifier")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save model")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"], help="Device to train on")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    datapoints = load_data(args.data)
    print(f"Loaded {len(datapoints)} valid molecules")

    # Stratified split into train/val
    labels = [dp.y[0] for dp in datapoints]
    train_data, val_data = train_test_split(
        datapoints, test_size=0.2, random_state=42, stratify=labels
    )

    train_dset = MoleculeDataset(train_data)
    val_dset = MoleculeDataset(val_data)

    train_loader = build_dataloader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dset, batch_size=args.batch_size, shuffle=False)

    # Build model (input_dim = 300 MP output + 9 condition dims)
    mp = BondMessagePassing(dropout=0.2)
    agg = NormAggregation()
    ffn = BinaryClassificationFFN(input_dim=mp.output_dim + 9, dropout=0.2)
    model = MPNN(message_passing=mp, agg=agg, predictor=ffn)

    # Callbacks: early stopping + best model checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    best_ckpt = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, "best"),
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.save_dir,
        accelerator=args.device,
        callbacks=[early_stop, best_ckpt],
        enable_progress_bar=True,
        logger=True,
    )
    trainer.fit(model, train_loader, val_loader)

    print(f"Training complete. Best checkpoint: {best_ckpt.best_model_path}")


if __name__ == "__main__":
    main()
