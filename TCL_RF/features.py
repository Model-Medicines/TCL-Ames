"""Shared feature engineering: ECFP4 fingerprints + Strain/S9 one-hot encodings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ECFP_RADIUS = 2
ECFP_N_BITS = 2048


def smiles_to_ecfp(smiles: str, radius: int = ECFP_RADIUS, n_bits: int = ECFP_N_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def one_hot_strain(strain_values: pd.Series, categories: list[str]) -> np.ndarray:
    encoded = np.zeros((len(strain_values), len(categories)), dtype=np.float32)
    category_index = {value: idx for idx, value in enumerate(categories)}
    for row_idx, value in enumerate(strain_values.astype(str)):
        col_idx = category_index.get(value)
        if col_idx is not None:
            encoded[row_idx, col_idx] = 1.0
    return encoded


def one_hot_s9(s9_values: pd.Series, categories: list[int]) -> np.ndarray:
    encoded = np.zeros((len(s9_values), len(categories)), dtype=np.float32)
    category_index = {value: idx for idx, value in enumerate(categories)}
    for row_idx, value in enumerate(s9_values.astype(int)):
        col_idx = category_index.get(value)
        if col_idx is not None:
            encoded[row_idx, col_idx] = 1.0
    return encoded


def build_feature_matrix(
    df: pd.DataFrame,
    strain_categories: list[str],
    s9_categories: list[int],
) -> np.ndarray:
    ecfp_matrix = np.vstack(
        [smiles_to_ecfp(smiles) for smiles in df["SMILES"].astype(str)]
    )
    strain_matrix = one_hot_strain(df["Strain"], strain_categories)
    s9_matrix = one_hot_s9(df["S9"], s9_categories)
    return np.hstack([ecfp_matrix, strain_matrix, s9_matrix])
