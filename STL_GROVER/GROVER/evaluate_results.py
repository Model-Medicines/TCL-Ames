"""
Aggregate and evaluate GROVER fine-tuning results across all STL variants.

Reads test_result.csv from each variant's results folder and computes
classification metrics. Outputs a summary CSV.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

VARIANTS = [
    "TA100_with_S9", "TA100_without_S9",
    "TA102_with_S9", "TA102_without_S9",
    "TA104_with_S9", "TA104_without_S9",
    "TA1535_with_S9", "TA1535_without_S9",
    "TA1537_with_S9", "TA1537_without_S9",
    "TA1538_with_S9", "TA1538_without_S9",
    "TA97_with_S9", "TA97_without_S9",
    "TA98_with_S9", "TA98_without_S9",
]

TASK_NAME = "ames_mutagenicity"


def evaluate_variant(variant):
    """Load test predictions and compute metrics for a single variant."""
    variant_dir = os.path.join(RESULTS_DIR, variant)
    test_csv = None
    for root, dirs, files in os.walk(variant_dir):
        if "test_result.csv" in files:
            test_csv = os.path.join(root, "test_result.csv")
            break
    if test_csv is None:
        test_csv = os.path.join(variant_dir, "test_result.csv")  # fallback for error msg
    if not os.path.exists(test_csv):
        print(f"  WARNING: {test_csv} not found, skipping {variant}")
        return None

    df = pd.read_csv(test_csv, index_col=0, header=[0, 1])
    probs = df[("preds", TASK_NAME)].values
    targets = df[("targets", TASK_NAME)].values.astype(int)
    binary_preds = (probs >= 0.5).astype(int)

    n_test = len(targets)
    n_pos = targets.sum()
    n_neg = n_test - n_pos

    auroc = roc_auc_score(targets, probs)
    ba = balanced_accuracy_score(targets, binary_preds)
    f1 = f1_score(targets, binary_preds, zero_division=0)
    mcc = matthews_corrcoef(targets, binary_preds)

    tn, fp, fn, tp = confusion_matrix(targets, binary_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Variant": variant,
        "N_test": n_test,
        "N_pos": int(n_pos),
        "N_neg": int(n_neg),
        "Pos_rate": round(n_pos / n_test, 3),
        "AUROC": round(auroc, 4),
        "Balanced_Acc": round(ba, 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    }


if __name__ == "__main__":
    print("Evaluating GROVER fine-tuning results...\n")

    rows = []
    for variant in VARIANTS:
        result = evaluate_variant(variant)
        if result:
            rows.append(result)
            print(f"  {variant}: AUROC={result['AUROC']:.4f}  BA={result['Balanced_Acc']:.4f}  "
                  f"Sens={result['Sensitivity']:.4f}  Spec={result['Specificity']:.4f}  "
                  f"F1={result['F1']:.4f}  MCC={result['MCC']:.4f}")

    if not rows:
        print("No results found. Run fine-tuning first.")
        exit(1)

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "summary_all_variants.csv")
    summary_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"Summary across {len(rows)} variants:")
    for metric in ["AUROC", "Balanced_Acc", "Sensitivity", "Specificity", "F1", "MCC"]:
        vals = summary_df[metric].values
        print(f"  {metric:15s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    print(f"\nSaved to: {out_path}")
