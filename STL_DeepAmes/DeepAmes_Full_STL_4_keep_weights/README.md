# DeepAmes Full STL (Keep Weights)

This folder contains two related workflows built on pre-featurized Ames datasets:

1. **Training pipeline**: train base models, select base learners, train DeepAmes+ weights `6..18`, and generate bootstrap reports.
2. **Inference-only pipeline**: load saved base/deep artifacts and run forward predictions without retraining.

## Conda Environment (Required)

Run this project in the **`deepames`** conda environment.

```bash
conda activate deepames
```

Tested environment snapshot (from `conda list -n deepames`):

- `python==3.7.3`
- `numpy==1.16.4`
- `pandas==0.24.2`
- `scikit-learn==0.21.2`
- `scipy==1.5.4`
- `tensorflow==1.14.0`
- `tensorflow-estimator==1.14.0`
- `keras==2.2.4`
- `h5py==2.10.0`
- `xgboost==1.6.2`
- `tqdm==4.67.2`
- `joblib==1.3.2`

Core dependencies used by scripts:

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow` (TF1-style API)
- `keras`
- `xgboost`
- `tqdm`
- `joblib`

Quick import check:

```bash
conda activate deepames
python -c "import numpy, pandas, sklearn, tensorflow, keras, xgboost, joblib, tqdm; print('deps OK')"
```

## Project Layout

### Training scripts

- `run_multi_dataset.py`: CLI runner for one dataset or all datasets from `Ready_Data/`
- `main.py`: core orchestration (`run_pipeline(...)`)
- `base_knn.py`, `base_lr.py`, `base_svm.py`, `base_rf.py`, `base_xgboost.py`: base learner training
- `select_base.py`: MCC-based base model selection
- `validation_predictions_combine.py`, `test_predictions_combine.py`: assemble stacked probability features
- `deepames_plus.py`: train/evaluate DeepAmes+ models across weights
- `generate_metrics_report.py`: bootstrap CI report generation
- `artifact_utils.py`: artifact persistence helpers

### Inference and post-processing scripts

- `inference_saved_models.py`: inference-only runner (single dataset/weight or batch)
- `combine.py`: combine trained `All_Results/*/result/test_class/*.csv` into `Final_Results/`
- `combine_inference_results.py`: combine `Inference_Results/*_inference.csv` into `Final_Results/`

### Key directories

- `Ready_Data/`: train/test featurized CSVs + `all_features.csv`
- `All_Results/`: default training outputs from `run_multi_dataset.py`
- `Saved_Base_Model_Artifacts/`: saved base artifact indexes grouped by dataset/model family
- `Saved_DeepAmes_Model_Weights/`: saved `weight_<6..18>.h5` models by dataset
- `Base_Artifact_Index_CSVs/`: selected-base index CSVs used by inference
- `Inference_Metadata_Artifacts/`: schema + validation-probability metadata used by inference
- `Inference_Results/`: per-dataset inference outputs
- `Final_Results/`: combined final CSV outputs

## Input Data Requirements

By default, the training runner expects:

- `Ready_Data/all_features.csv` with one column named `feature`
- `Ready_Data/Train_Data_Featurized/*_Train_Val_mold2.csv`
- `Ready_Data/Test_Data_Featurized/*_Test_mold2.csv`

Expected train/test CSV columns:

- Identifier columns: `gmtamesQSAR_ID`, `SMILES` (kept in outputs, not used as model features)
- Target column: `label` (`0/1`)
- Descriptor columns: `D001 ... D777` (must match `all_features.csv`)

Dataset naming convention:

- Train file: `<DATASET_NAME>_Train_Val_mold2.csv`
- Test file: `<DATASET_NAME>_Test_mold2.csv`
- Example dataset name: `TA98_without_S9`

## Run Training Pipeline

Run from this folder:

```bash
cd /home/ubuntu/Desktop/DeepAmes_Full_STL_4_keep_weights
conda activate deepames
```

### Run one dataset

```bash
python run_multi_dataset.py --dataset TA98_without_S9
```

### Run all discovered datasets

```bash
python run_multi_dataset.py
```

### Optional arguments

```bash
python run_multi_dataset.py --base-dir ./Ready_Data --output-base ./All_Results
```

## Training Outputs

By default, each dataset is written to:

- `All_Results/results_<DATASET_NAME>/`

Main outputs per dataset include:

- `base/`: base model predictions and performance tables
- `probabilities_output/validation_probabilities_<DATASET_NAME>.csv`
- `probabilities_output/test_probabilities_<DATASET_NAME>.csv`
- `result/validation_class/` and `result/test_class/` per DeepAmes weight
- `result/validation_performance/` and `result/test_performance/`
- `DeepAmes_models/weight_<6..18>.h5`
- `artifacts/selected_base_artifacts.csv`
- `artifacts/schema/validation_schema_<DATASET_NAME>.json`
- `artifacts/deepames/weight_<N>_manifest.json` and `weight_<N>_scaler.joblib`
- `artifacts/run_manifest_<DATASET_NAME>.json`
- `metrics_report_<DATASET_NAME>.txt`

## Run Inference-Only Pipeline

`inference_saved_models.py` does not retrain models. It loads saved artifacts and predicts test-set outputs.

### Single dataset, single weight

```bash
python inference_saved_models.py \
  --dataset-name TA98_without_S9 \
  --weight 6
```

Default output:

- `Inference_Results/TA98_without_S9_weight6_inference.csv`

### Batch mode (all discovered datasets, weight range)

```bash
python inference_saved_models.py \
  --run-all \
  --weights-start 6 \
  --weights-end 18
```

Optional batch scope/output:

```bash
python inference_saved_models.py \
  --run-all \
  --datasets TA98_without_S9,TA98_with_S9 \
  --weights-start 6 \
  --weights-end 6 \
  --batch-output-dir ./Inference_Results
```

Useful inference flags:

- `--threshold` (default `0.65`)
- `--base-jobs` (parallel base-model `predict_proba`; default auto-capped)
- `--saved-base-root`, `--saved-deep-root`
- `--base-artifact-index-root`, `--inference-metadata-root`
- path overrides such as `--features-path`, `--test-data-path`, `--deep-model-path`, `--output-csv`

## Combine Outputs Across Datasets

### Combine trained test-class outputs

```bash
python combine.py
```

Generates files like:

- `Final_Results/weight6_DeepAmes_predictions.csv`
- ...
- `Final_Results/weight18_DeepAmes_predictions.csv`

### Combine inference outputs

```bash
python combine_inference_results.py
```

Generates files like:

- `Final_Results/weight6_DeepAmes_inference_predictions.csv`
- (and additional weights when corresponding inference files exist)

## Model Selection Notes

- Base models are trained with fixed hyperparameter index choices (`var`) in each base script.
- `select_base.py` keeps base learners in the central MCC band (between 5th and 95th percentile).
- DeepAmes+ evaluates class-weight values from `6` to `18`.
- Metrics reports rank models by MCC and include bootstrap 90% confidence intervals.

## Re-run Metrics Report Only

If results already exist, regenerate report with:

```bash
python generate_metrics_report.py \
  --result-dir ./All_Results/results_TA98_without_S9 \
  --dataset-name TA98_without_S9
```

## Troubleshooting

- **Missing input pair**: confirm both train and test files exist with matching dataset stem.
- **Feature mismatch errors**: confirm `all_features.csv` matches descriptor columns in train/test files.
- **Missing saved inference artifacts**: verify dataset-specific files exist under `Saved_Base_Model_Artifacts`, `Saved_DeepAmes_Model_Weights`, `Base_Artifact_Index_CSVs`, and `Inference_Metadata_Artifacts`.
- **TensorFlow/Keras issues**: this code uses TF1-style APIs; run in `deepames`.
- **CPU over-subscription during inference**: lower `--base-jobs` and optionally set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.
- **Long runtime**: training all datasets across base models and 13 DeepAmes weights can take substantial time.
