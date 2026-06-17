# TCL-Ames

Code and data for the paper:

> **AmesNet: A Task-Conditioned Deep Learning Model with Enhanced Sensitivity and Generalization in Ames Mutagenicity Prediction**
>
> Tyler Umansky, Virgil Woods, Sean M. Russell, Daniel Haders
>
> bioRxiv 2025. DOI: [10.1101/2025.03.20.644379](https://doi.org/10.1101/2025.03.20.644379)

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `STL_ChemProp/` | Single-Task Learning (STL) ChemProp models — training data, checkpoints, per-strain predictions, and combined outputs |
| `STL_DeepAmes/` | Single-Task Learning (STL) DeepAmes models |
| `STL_GROVER/` | Single-Task Learning (STL) GROVER models — per-strain training data, processed inputs, and predictions |
| `STL_RF/` | Single-Task Learning (STL) random forest models — per-strain `.joblib` models, Lui and foil test predictions |
| `STL_uMTL_gMTL_MLP/` | Single-Task Learning (STL), Ungrouped Multitask Learning (uMTL), and Grouped Multitask Learning (gMTL) MLP models |
| `TCL_ChemProp/` | Task-Conditioned Learning (TCL) ChemProp models |
| `TCL_GROVER/` | Task-Conditioned Learning (TCL) GROVER models |
| `TCL_RF/` | Task-Conditioned Learning (TCL) random forest models — training scripts, outputs, and test-set predictions |
| `Ames Bootstrap Analysis/` | Bootstrap statistical analysis across all model predictions |

### `STL_DeepAmes/`

| Subdirectory | Description |
|--------------|-------------|
| `DeepAmes_Featurize_E_to_E/` | End-to-end preprocessing and MOLD2 featurization pipeline for strain/S9-specific DeepAmes inputs |
| `DeepAmes_Full_STL_4_keep_weights/` | Full STL DeepAmes training and inference pipeline, including saved DeepAmes weights, inference results, and bootstrap reports |

See the README files inside each subdirectory for workflow details.

### `Ames Bootstrap Analysis/`

| Subdirectory / File | Description |
|---------------------|-------------|
| `Raw_Prediction_Data/Lui_Test_Preds/` | Per-model prediction CSVs for the Lui test set |
| `Raw_Prediction_Data/Foil_Test_Preds/` | Combined foil-set prediction CSVs |
| `Bootstrapped_Data_95_CI/` | Bootstrap confidence interval results |
| `composite_weighted_bootstrap_analysis.py` | Script to compute sample-size weighted, task-averaged metrics with bootstrap CIs |

## Model Weights

Most training data, code, and predictions are included directly in this repository. However, several large model weight files exceed GitHub's file size limits and are **not** included here:

| Excluded from GitHub | Location | Reason |
|----------------------|----------|--------|
| GROVER `.pt` checkpoints | `STL_GROVER/GROVER/pretrained_models/`, `TCL_GROVER/GROVER/pretrained_models/` | >100 MB per file |
| TCL RF combined model | `TCL_RF/models/rf_ecfp_strain_s9.joblib` | >100 MB |
| DeepAmes base model artifacts | `STL_DeepAmes/DeepAmes_Full_STL_4_keep_weights/Saved_Base_Model_Artifacts/` | ~51 GB total |
| MLP checkpoints | `STL_uMTL_gMTL_MLP/output/` | Hosted externally |

These files are hosted on Hugging Face:

**https://huggingface.co/Model-Medicines-Inc/TCL-Ames**

The Hugging Face repository mirrors the same directory structure as this repo. To use the pretrained weights, download them and place them in the corresponding directories.

**Included in this repo:** STL RF per-strain `.joblib` models (`STL_RF/models/`, 53–72 MB each) and DeepAmes model weights (`Saved_DeepAmes_Model_Weights/`).
