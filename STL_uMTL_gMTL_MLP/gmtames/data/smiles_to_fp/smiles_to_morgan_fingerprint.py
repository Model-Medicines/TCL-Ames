"""
smiles_to_morgan_fingerprint.py

Compute 1024-bit Morgan fingerprints (radius=2) from SMILES.
Matches the featurization used in:
  Lui et al., Chem. Res. Toxicol. 2023, 36, 1248−1254
  "Binary Morgan fingerprints, a reimplementation of extended connectivity
   fingerprints, with length 1,024 bits and radius of 2 bond lengths."

Can take a CSV file (like gmtamesQSAR_endpoints_scaffold.csv but with a SMILES column),
featurize each unique gmtamesQSAR_ID, and by default write:
  gmtames/data/master_datasets/gmtamesQSAR_fingerprints.csv
  gmtames/data/master_datasets/gmtamesQSAR_endpoints_scaffold.csv (input CSV without SMILES)
"""

import argparse
import csv
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_DATASETS_DIR = _SCRIPT_DIR.parent / 'master_datasets'
DEFAULT_FINGERPRINTS_CSV = MASTER_DATASETS_DIR / 'gmtamesQSAR_fingerprints.csv'
DEFAULT_ENDPOINTS_SCAFFOLD_CSV = MASTER_DATASETS_DIR / 'gmtamesQSAR_endpoints_scaffold.csv'

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("Error: RDKit is required. Install with: conda install -c conda-forge rdkit", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None, **kwargs):
        return iterable

# Paper parameters
N_BITS = 1024
RADIUS = 2

# Cache: (smiles_normalized, n_bits, radius) -> list of int or None (invalid)
_morgan_fp_cache = {}


def _cache_key(smiles, n_bits, radius):
    s = smiles.strip() if isinstance(smiles, str) else ""
    return (s, n_bits, radius)


def clear_fingerprint_cache():
    """Clear the SMILES -> fingerprint cache (e.g. for memory or fresh run)."""
    _morgan_fp_cache.clear()


def smiles_to_morgan_fingerprint(smiles, n_bits=N_BITS, radius=RADIUS, use_cache=True):
    """
    Compute binary Morgan fingerprint from a SMILES string.
    Results are cached by (SMILES, n_bits, radius) so the same SMILES is not recomputed.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    n_bits : int
        Fingerprint length (default 1024, per paper).
    radius : int
        Morgan radius in bond lengths (default 2, per paper).
    use_cache : bool
        If True (default), use and update the cache. Set False to force recompute.

    Returns
    -------
    list of int
        Length n_bits, values 0 or 1. None if SMILES is invalid.
    """
    if use_cache:
        key = _cache_key(smiles, n_bits, radius)
        if key in _morgan_fp_cache:
            return _morgan_fp_cache[key]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result = None
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        result = [int(b) for b in fp.ToBitString()]
    if use_cache:
        _morgan_fp_cache[key] = result
    return result


def smiles_to_morgan_row(smiles, mol_id=None, n_bits=N_BITS, radius=RADIUS):
    """
    Compute Morgan fingerprint as a row compatible with gmtamesQSAR_fingerprints.csv.

    Parameters
    ----------
    smiles : str
        SMILES string.
    mol_id : str, optional
        Molecule ID (e.g. gmtamesQSAR_0001). If None, not included.
    n_bits : int
        Fingerprint length (default 1024).
    radius : int
        Morgan radius (default 2).

    Returns
    -------
    dict or None
        Keys: 'gmtamesQSAR_ID' (if mol_id given), 'Bit 1', 'Bit 2', ... 'Bit 1024'.
        None if SMILES is invalid.
    """
    bits = smiles_to_morgan_fingerprint(smiles, n_bits=n_bits, radius=radius)
    if bits is None:
        return None
    row = {}
    if mol_id is not None:
        row['gmtamesQSAR_ID'] = mol_id
    for i, b in enumerate(bits, start=1):
        row[f'Bit {i}'] = b
    return row


def csv_to_fingerprints(input_path, output_path, id_col='gmtamesQSAR_ID', smiles_col='SMILES',
                        n_bits=N_BITS, radius=RADIUS, endpoints_path=None):
    """
    Read a CSV with gmtamesQSAR_ID and SMILES columns, compute Morgan fingerprints,
    and write output in gmtamesQSAR_fingerprints.csv format.

    Parameters
    ----------
    input_path : str or Path
        Input CSV path (e.g. endpoints-like file with SMILES column).
    output_path : str or Path
        Output CSV path (same format as gmtamesQSAR_fingerprints.csv).
    id_col : str
        Column name for molecule ID (default gmtamesQSAR_ID).
    smiles_col : str
        Column name for SMILES (default SMILES).
    n_bits : int
        Fingerprint length (default 1024).
    radius : int
        Morgan radius (default 2).
    endpoints_path : str or Path, optional
        If set, write the full input table without ``smiles_col`` for ``generateBaseDatasets``.
    """
    if pd is None:
        raise ImportError("pandas is required for CSV file mode. Install with: pip install pandas")

    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    if id_col not in df.columns:
        raise ValueError(f"Input CSV must have column '{id_col}'")
    if smiles_col not in df.columns:
        raise ValueError(f"Input CSV must have column '{smiles_col}'")

    if endpoints_path is not None:
        endpoints_path = Path(endpoints_path)
        endpoints_path.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=[smiles_col]).to_csv(endpoints_path, index=False)
        print(f"Wrote endpoints table ({len(df)} rows) to {endpoints_path}", file=sys.stderr)

    # One row per unique ID: keep first occurrence (in case of duplicates across strain/endpoint rows)
    id_smiles = df.groupby(id_col, as_index=False)[smiles_col].first()
    n_unique = len(id_smiles)

    # Build output rows (cache reused when the same SMILES appears for different IDs)
    header = [id_col] + [f'Bit {i}' for i in range(1, n_bits + 1)]
    rows = []
    failed = 0
    cache_size_before = len(_morgan_fp_cache)
    for _, r in id_smiles.iterrows():
        mol_id = r[id_col]
        smi = r[smiles_col]
        if pd.isna(smi) or not str(smi).strip():
            print(f"Warning: empty SMILES for {mol_id}, skipping", file=sys.stderr)
            failed += 1
            continue
        row_dict = smiles_to_morgan_row(str(smi).strip(), mol_id=mol_id, n_bits=n_bits, radius=radius)
        if row_dict is None:
            print(f"Warning: invalid SMILES for {mol_id}: {smi}", file=sys.stderr)
            failed += 1
            continue
        rows.append([row_dict[k] for k in header])

    # Write in same format as gmtamesQSAR_fingerprints.csv (quoted header, ID quoted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    cache_size_after = len(_morgan_fp_cache)
    unique_computed = cache_size_after - cache_size_before
    print(f"Wrote {len(rows)} fingerprints to {output_path}", file=sys.stderr)
    print(f"Cache: {unique_computed} unique SMILES computed, {n_unique - unique_computed} lookups reused", file=sys.stderr)
    if failed:
        print(f"Skipped {failed} rows (empty or invalid SMILES)", file=sys.stderr)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Compute 1024-bit Morgan fingerprint (radius=2) from SMILES or from a CSV file.'
    )
    parser.add_argument(
        'input',
        nargs='?',
        default=None,
        help='SMILES string, or path to CSV file with gmtamesQSAR_ID and SMILES columns',
    )
    parser.add_argument(
        '-i', '--input-file',
        dest='input_file',
        default=None,
        help='Path to input CSV (like gmtamesQSAR_endpoints_scaffold.csv but with SMILES column)',
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help=f'Path to fingerprints CSV (default: {DEFAULT_FINGERPRINTS_CSV}) in CSV input mode.',
    )
    parser.add_argument(
        '--endpoints-output',
        default=None,
        help=f'Path to endpoints CSV without SMILES (default: {DEFAULT_ENDPOINTS_SCAFFOLD_CSV}). '
        'Ignored with --no-endpoints.',
    )
    parser.add_argument(
        '--no-endpoints',
        action='store_true',
        help='Do not write gmtamesQSAR_endpoints_scaffold.csv (fingerprints only).',
    )
    parser.add_argument(
        '--id-column',
        default='gmtamesQSAR_ID',
        help='Name of ID column in input CSV (default: gmtamesQSAR_ID)',
    )
    parser.add_argument(
        '--smiles-column',
        default='SMILES',
        help='Name of SMILES column in input CSV (default: SMILES)',
    )
    parser.add_argument(
        '--id',
        default=None,
        help='Optional molecule ID when input is a single SMILES string',
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Print single SMILES as CSV row (ID, Bit 1, ..., Bit 1024)',
    )
    parser.add_argument(
        '--n-bits',
        type=int,
        default=N_BITS,
        help=f'Number of bits (default {N_BITS})',
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=RADIUS,
        help=f'Morgan radius (default {RADIUS})',
    )
    args = parser.parse_args()

    # File mode: input CSV -> output fingerprints CSV
    infile = args.input_file or args.input
    if infile and (Path(infile).suffix.lower() == '.csv' or args.input_file):
        infile = Path(args.input_file or args.input)
        if not infile.exists():
            print(f"Error: input file not found: {infile}", file=sys.stderr)
            sys.exit(1)
        out_fp = Path(args.output) if args.output else DEFAULT_FINGERPRINTS_CSV
        if args.no_endpoints:
            ep_path = None
        else:
            ep_path = Path(args.endpoints_output) if args.endpoints_output else DEFAULT_ENDPOINTS_SCAFFOLD_CSV
        csv_to_fingerprints(
            infile,
            out_fp,
            id_col=args.id_column,
            smiles_col=args.smiles_column,
            n_bits=args.n_bits,
            radius=args.radius,
            endpoints_path=ep_path,
        )
        return

    # Single SMILES or stdin mode
    if args.input is not None:
        inputs = [(args.id or 'stdin', args.input)]
    else:
        inputs = []
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            mol_id = parts[0] if len(parts) > 1 else None
            smi = parts[-1]
            inputs.append((mol_id, smi))

    if args.csv and inputs:
        header = ['gmtamesQSAR_ID'] + [f'Bit {i}' for i in range(1, args.n_bits + 1)]
        print(','.join(header))

    for mol_id, smi in inputs:
        if args.csv:
            row = smiles_to_morgan_row(smi, mol_id=mol_id, n_bits=args.n_bits, radius=args.radius)
            if row is None:
                print(f"Invalid SMILES: {smi}", file=sys.stderr)
                continue
            header = list(row.keys())
            print(','.join(str(row[k]) for k in header))
        else:
            bits = smiles_to_morgan_fingerprint(smi, n_bits=args.n_bits, radius=args.radius)
            if bits is None:
                print(f"Invalid SMILES: {smi}", file=sys.stderr)
                continue
            print(''.join(str(b) for b in bits))


if __name__ == '__main__':
    main()
