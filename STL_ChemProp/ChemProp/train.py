"""Vanilla ChemProp v2 training script for binary classification."""

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


def load_data(csv_path: str) -> list[MoleculeDatapoint]:
    """Load a CSV with SMILES and Endpoint columns into MoleculeDatapoints."""
    df = pd.read_csv(csv_path)
    label_map = {"Negative": 0, "Positive": 1}
    datapoints = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            continue
        y = np.array([label_map[row["Endpoint"]]], dtype=np.float32)
        datapoints.append(MoleculeDatapoint(mol=mol, y=y))
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

    # Build model
    mp = BondMessagePassing(dropout=0.2)
    agg = NormAggregation()
    ffn = BinaryClassificationFFN(dropout=0.2)
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
