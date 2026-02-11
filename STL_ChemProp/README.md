# STL ChemProp

Single-task learning (STL) Ames mutagenicity prediction using [ChemProp v2](https://github.com/chemprop/chemprop) message-passing neural networks. A separate binary classification model is trained for each Ames strain/S9 variant (e.g. TA100 with S9, TA98 without S9).

## Project Structure

```
STL_ChemProp/
├── Data/
│   ├── Dataset_Variants_Train_Val/   # Training/validation CSVs (one per variant)
│   └── Dataset_Variants_Test/        # Test CSVs (one per variant)
└── ChemProp/
    ├── train.py            # Train a single ChemProp model
    ├── predict.py          # Run inference with a trained checkpoint
    ├── run_all.py                  # Train + predict across all 16 variants
    ├── combine_stl_predictions.py  # Merge per-variant predictions into one CSV
    ├── requirements.txt
    ├── checkpoints/                # Saved model checkpoints
    ├── predictions/                # Per-variant prediction CSVs
    └── predictions_combined/       # Combined summary CSV
```

## Setup

```bash
pip install -r ChemProp/requirements.txt
```

## Usage

**Train and predict all variants:**

```bash
python ChemProp/run_all.py
```

**Train a single variant:**

```bash
python ChemProp/train.py --data Data/Dataset_Variants_Train_Val/TA98_with_S9_Train_Val.csv \
    --save-dir ChemProp/checkpoints/TA98_with_S9
```

**Predict with a trained model:**

```bash
python ChemProp/predict.py --data Data/Dataset_Variants_Test/TA98_with_S9_Test.csv \
    --checkpoint ChemProp/checkpoints/TA98_with_S9/best/best-*.ckpt \
    --output ChemProp/predictions/TA98_with_S9_predictions.csv
```

**Combine all predictions into a single CSV:**

```bash
python ChemProp/combine_stl_predictions.py
```

This reads all per-variant CSVs from `ChemProp/predictions/` and writes a combined summary to `ChemProp/predictions_combined/STL_ChemProp_predictions.csv`.

## Data Format

Input CSVs require `SMILES` and `Endpoint` columns, where `Endpoint` is `Positive` or `Negative`.
