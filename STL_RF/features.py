"""ECFP4 fingerprint features for single-task random forest models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ECFP_RADIUS = 2
ECFP_N_BITS = 2048


def smiles_to_ecfp(
    smiles: str,
    radius: int = ECFP_RADIUS,
    n_bits: int = ECFP_N_BITS,
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(str(smiles)) is not None


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        [smiles_to_ecfp(smiles) for smiles in df["SMILES"].astype(str)]
    )
