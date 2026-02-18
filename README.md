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
| `STL_ChemProp/` | Single-Task Learning (STL) ChemProp models |
| `STL_DeepAmes/` | Single-Task Learning (STL) DeepAmes models |
| `STL_GROVER/` | Single-Task Learning (STL) GROVER models |
| `STL_uMTL_gMTL_MLP/` | Single-Task Learning (STL), Ungrouped Multitask Learning (uMTL), and Grouped Multitask Learning (gMTL) MLP models |
| `TCL_ChemProp/` | Task-Conditioned Learning (TCL) ChemProp models |
| `TCL_GROVER/` | Task-Conditioned Learning (TCL) GROVER models |
| `Ames Bootstrap Analysis/` | Bootstrap statistical analysis |

## Model Weights

All training data, code, and predictions are included directly in this repository. However, the trained model weight files (`.pt`) for `STL_GROVER/`, `TCL_GROVER/`, and `STL_uMTL_gMTL_MLP/` exceed GitHub's file size limits and are hosted on Hugging Face:

**https://huggingface.co/Model-Medicines-Inc/TCL-Ames**

The Hugging Face repository mirrors the same directory structure as this repo. To use the pretrained weights, download them and place them in the corresponding directories.
