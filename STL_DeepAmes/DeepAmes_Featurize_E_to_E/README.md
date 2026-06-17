# DeepAmes Featurize (E-to-E)

This directory contains the end-to-end preprocessing and featurization pipeline used to build strain/S9-specific DeepAmes input files from master Ames datasets.

The workflow:
1. Split each master dataset into 16 dataset variants (`8 strains x 2 S9 conditions`)
2. Generate MOLD2 descriptors from SMILES
3. Build a consolidated descriptor-name list (`all_features.csv`)

Across both master files (`Train_Val` and `Test`), a full run produces 32 variant CSVs total.

## Directory Layout

Current structure in this project:

- `Fixed_Leakage_Master_Data/`
  - `train_val_ood_master_df.csv`
  - `test_ood_master_df.csv`
- `STL_Data/`
  - `Dataset_Variants_Train_Val/` (16 split files)
  - `Dataset_Variants_Test/` (16 split files)
- `Train_Data_Featurized/` (MOLD2 output files)
- `Test_Data_Featurized/` (MOLD2 output files)
- `create_variants.py`
- `generate_mold2_multiprocessing.py`
- `create_all_features.py`
- `all_features.csv`

## Input and Output Schemas

### Master input files

Expected columns:
- `gmtamesQSAR_ID`
- `SMILES`
- `Strain`
- `S9`
- `Endpoint` (`Positive` / `Negative`)

### Featurized output files

Columns are:
- `gmtamesQSAR_ID`
- `SMILES`
- `label` (current script behavior: `Positive -> 1`, any other value -> `0`)
- `D001 ... D777` (MOLD2 descriptors)

## Requirements

### Conda environment (required)

This pipeline **must be run in the `deepames_featurize` conda environment**.
Do not run these scripts from your base environment.

Activate before running any script:

```bash
conda activate deepames_featurize
```

### Dependencies in `deepames_featurize`

Required package versions used in this project:

- `python==3.7.3`
- `pandas==1.3.5`
- `rdkit==2023.3.2`
- `mold2-pywrapper==0.0.3`
- `numpy==1.21.6`

Direct Python dependencies imported by scripts:

- `pandas`
- `rdkit`
- `mold2-pywrapper`

(`argparse`, `glob`, `os`, `time`, `warnings`, and `multiprocessing` are Python standard-library modules and do not require separate installation.)

### Create the environment from scratch

Create/activate your environment, then install dependencies:

```bash
conda create -n deepames_featurize python=3.7.3 -y
conda activate deepames_featurize
conda install -c conda-forge rdkit==2023.3.2 -y
pip install numpy==1.21.6 pandas==1.3.5 mold2-pywrapper==0.0.3
```

If your RDKit solve differs on your machine, use:

```bash
conda install -c conda-forge rdkit
```

Quick verification (shows the env packages are importable):

```bash
conda activate deepames_featurize
python -c "import pandas as pd; import rdkit; from Mold2_pywrapper import Mold2; print('pandas', pd.__version__); print('rdkit', rdkit.__version__); print('Mold2 import OK')"
```

## Pipeline Usage

Run all commands from this directory (with `deepames_featurize` already activated):

```bash
cd /home/ubuntu/Desktop/DeepAmes_Featurize_E_to_E
```

### 1) Create variant datasets

`create_variants.py` currently uses hard-coded values:
- `name_of_files = "Train_Val"`
- source file: `Fixed_Leakage_Master_Data/train_val_ood_master_df.csv`
- output dir: `STL_Data/Dataset_Variants_Train_Val`

Important:
- The script uses hard-coded absolute paths rooted at `/home/ubuntu/Desktop/DeepAmes_Featurize_E_to_E`.
- If you move this project, update those paths in the script before running.

Run:

```bash
python create_variants.py
```

To generate test variants, update the script values similarly (for example, `name_of_files = "Test"` and source `test_ood_master_df.csv`) and run again.

### 2) Generate MOLD2 descriptors

`generate_mold2_multiprocessing.py` runs in batch mode over every CSV in:
- input: `STL_Data/Dataset_Variants_Test`
- output: `Test_Data_Featurized`

Run:

```bash
python generate_mold2_multiprocessing.py
```

Notes:
- CPU usage defaults to all available cores (`os.cpu_count()`).
- The script skips invalid SMILES rows and reports warnings.
- Output files are named with `_mold2.csv`.
- The script currently maps labels as: `1` if `Endpoint == "Positive"`, else `0`.

Important:
- This script also uses hard-coded absolute paths rooted at `/home/ubuntu/Desktop/DeepAmes_Featurize_E_to_E`.
- If you move this project, update `input_dir` and `output_dir` in the script.

If you need train/val featurization, change `input_dir` and `output_dir` in the script to the Train_Val variant and target featurized directory.

### 3) Build unified feature list

Create `all_features.csv` from both featurized folders:

```bash
python create_all_features.py
```

Optional flags:

```bash
python create_all_features.py --base-dir /path/to/DeepAmes_Featurize_E_to_E --output-file /path/to/all_features.csv
```

## Script Details

- `create_variants.py`
  - Splits one master file into 16 subsets by `Strain` and `S9`.
- `generate_mold2_multiprocessing.py`
  - Converts `SMILES` to RDKit molecules and computes MOLD2 descriptors in parallel.
  - Produces model-ready CSVs with `label` plus descriptor columns.
  - Current label logic is `1` for `Positive`, otherwise `0`.
- `create_all_features.py`
  - Reads featurized CSV headers from `Train_Data_Featurized` and `Test_Data_Featurized`.
  - Writes one-column `all_features.csv` with descriptor names.

## Quick Validation

After each stage, confirm expected file counts:

- `STL_Data/Dataset_Variants_*`: 16 files each
- `Train_Data_Featurized` / `Test_Data_Featurized`: 16 `_mold2.csv` files each
- `all_features.csv`: one column named `feature`, usually listing `D001 ... D777`

## Troubleshooting

- **No files found**: verify source/input directories in each script.
- **Unexpected label values**: `generate_mold2_multiprocessing.py` currently maps any non-`Positive` endpoint to `0`.
- **RDKit import errors**: confirm your environment and RDKit installation.
- **MOLD2 wrapper errors**: ensure `Mold2_pywrapper` is installed and callable from the active Python environment.
- **Slow runtime**: reduce `n_cpus` if needed by editing the script call or function argument.
