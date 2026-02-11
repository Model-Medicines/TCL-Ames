# gmtames Reproduction Package

This folder contains everything needed to reproduce **Table 2** from the gmtames paper:

> **Averaged Out-of-Domain Test Set Performance Across 16 S. typhimurium ± S9 Strain Tasks, 
> Predicted by Neural Networks Developed with Single Task (STL), Ungrouped Multitask (uMTL), 
> Grouped Multitask (gMTL), and Overall Learning Architectures**

## Expected Results (Table 2)

| Deep learning architecture | Balanced accuracy | Sensitivity | Specificity | ROC AUC |
|---------------------------|-------------------|-------------|-------------|---------|
| STL | 0.700 (0.592–0.813) | 0.448 (0.238–0.675) | 0.951 (0.913–0.979) | 0.799 (0.664–0.915) |
| uMTL | 0.716 (0.602–0.828) | 0.542 (0.317–0.749) | 0.890 (0.834–0.943) | 0.823 (0.713–0.913) |
| gMTL | 0.745 (0.628–0.858) | 0.575 (0.347–0.788) | 0.915 (0.863–0.957) | 0.832 (0.721–0.923) |
| Overall | 0.756 (0.638–0.865) | 0.599 (0.367–0.801) | 0.913 (0.861–0.958) | 0.832 (0.720–0.923) |

*95% (shown) and 83% confidence intervals were averaged across the 16 strain tasks.*

---

## Directory Structure

```
gmtames_reproduction_filtered_data/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_experiment.sh         # Main execution script
└── gmtames/                  # Source code package
    ├── __init__.py
    ├── __main__.py           # Entry point & CLI
    ├── data.py               # Data loading & preprocessing
    ├── nn.py                 # Neural network architectures & training
    ├── mtg.py                # Mechanistic Task Grouping experiment runner
    ├── results.py            # Bootstrapping & metrics calculation
    ├── logging.py            # Logging utilities
    └── data/
        ├── smiles_to_fp/
        │   ├── gmtamesQSAR_endpoints_scaffold_with_smiles.csv  # Source: endpoints + SMILES
        │   ├── smiles_to_morgan_fingerprint.py                 # Generate fingerprints from SMILES
        │   ├── gmtamesQSAR_fingerprints.csv                    # 1024-bit Morgan (run script to create)
        │   └── gmtamesQSAR_endpoints_scaffold.csv              # Strain labels (run script to create)
        └── base_datasets/
            └── scaffold/     # Pre-split train/val/test datasets per strain (generated)
```

---

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- numpy
- pandas
- torch (PyTorch)
- scikit-learn
- optuna
- matplotlib

---

## Running the Reproduction

### Quick Start

**Prerequisite:** If you only have `gmtamesQSAR_endpoints_scaffold_with_smiles.csv` in `gmtames/data/smiles_to_fp/`, generate the fingerprint and endpoint files first (see Troubleshooting).

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

This will:
1. Run exploratory data analysis
2. Train 16 STL models (one per strain)
3. Train 1 uMTL model (all 16 strains)
4. Train 8 gMTL models (mechanistic groupings)
5. Calculate bootstrapped metrics with 95% confidence intervals

**Note:** Training takes several hours on CPU. Use `--device cuda:0` for GPU acceleration.

### Manual Execution

```bash
# Step 1: Exploratory data analysis
python -m gmtames data --describeBaseDatasets --correlateBaseDatasets \
    --testsplit scaffold --output mtg_experiment_scaffold

# Step 2: Train individual models (example for TA100)
python -m gmtames mtg --tasks TA100 --testsplit scaffold \
    --output mtg_experiment_scaffold --device cpu

# Step 3: Calculate results
python -m gmtames results mtg_experiment_scaffold
```

---

## Methodology

### Featurization
- **Morgan fingerprints** (ECFP) computed with RDKit
- **1024-bit** binary vectors

### Data Split
- **Scaffold split** for out-of-domain evaluation
- Train/Validation/Test partitions per strain

### Model Architecture
- Feedforward neural networks with ReLU activations
- Hyperparameter optimization via Optuna grid search:
  - Layers: 2, 3, or 4
  - Architecture styles: linear, top, bottom, outer, inner
  - Learning rate: 0.001 or 0.0001
  - Epochs: 100, 150, or 200

### Evaluation
- **1,000 bootstrap iterations** per strain task
- Stratified resampling (preserves positive/negative ratio)
- **95% and 83% confidence intervals** via percentile method
- Metrics: Balanced Accuracy, Sensitivity, Specificity, ROC AUC

### 16 Strain Tasks
| Strain | Mutation Type | S9 Activation |
|--------|---------------|---------------|
| TA100, TA102, TA104, TA1535 | Base-pair substitution | ± |
| TA1537, TA1538, TA97, TA98 | Frameshift | ± |

---

## Output Files

After running, results are saved to `output/mtg_experiment_scaffold/test_results/`:

| File | Description |
|------|-------------|
| `gmtames_curated_averaged_results.csv` | **Table 2 metrics** (architecture-averaged) |
| `gmtames_curated_task_results.csv` | Per-strain results for each architecture |
| `gmtames_full_results.csv` | Complete bootstrap statistics with all CI bounds |
| `gmtames_curated_task_groupings.csv` | gMTL grouping assignments |

---

## Citation

If you use this code, please cite the original gmtames paper.

---

## Troubleshooting

**Out of memory:** Reduce batch size or use CPU (`--device cpu`)

**CUDA not available:** Install PyTorch with CUDA support or use CPU

**Missing master data in `gmtames/data/smiles_to_fp/`:** Generate fingerprints and endpoints from the CSV with SMILES:
```bash
cd gmtames/data/smiles_to_fp
python smiles_to_morgan_fingerprint.py --input-file gmtamesQSAR_endpoints_scaffold_with_smiles.csv --output gmtamesQSAR_fingerprints.csv
```
This creates `gmtamesQSAR_fingerprints.csv` and `gmtamesQSAR_endpoints_scaffold.csv` in that directory.

**Missing base datasets:** Run data generation first:
```bash
python -m gmtames data --generateBaseDatasets --testsplit scaffold --output mtg_experiment_scaffold
```
