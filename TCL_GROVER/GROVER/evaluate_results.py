"""
Evaluate Task-Conditioned GROVER results.

Reads test_result.csv from the results folder and computes metrics
both overall and broken down by strain+S9 combination.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")

TASK_NAME = "ames_mutagenicity"


def compute_metrics(targets, probs):
    """Compute classification metrics from targets and predicted probabilities."""
    binary_preds = (probs >= 0.5).astype(int)
    n_test = len(targets)
    n_pos = int(targets.sum())

    metrics = {"N_test": n_test, "N_pos": n_pos, "Pos_rate": round(n_pos / n_test, 3)}

    if n_pos == 0 or n_pos == n_test:
        # Can't compute AUC with only one class
        metrics.update({"AUROC": float("nan"), "Balanced_Acc": float("nan"),
                        "Sensitivity": float("nan"), "Specificity": float("nan"),
                        "F1": float("nan"), "MCC": float("nan")})
        return metrics

    metrics["AUROC"] = round(roc_auc_score(targets, probs), 4)
    metrics["Balanced_Acc"] = round(balanced_accuracy_score(targets, binary_preds), 4)
    metrics["F1"] = round(f1_score(targets, binary_preds, zero_division=0), 4)
    metrics["MCC"] = round(matthews_corrcoef(targets, binary_preds), 4)

    tn, fp, fn, tp = confusion_matrix(targets, binary_preds, labels=[0, 1]).ravel()
    metrics["Sensitivity"] = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
    metrics["Specificity"] = round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4)

    return metrics


if __name__ == "__main__":
    print("Evaluating Task-Conditioned GROVER results...\n")

    # Load test predictions (search recursively since GROVER saves under fold_X/)
    test_csv = None
    for root, dirs, files in os.walk(RESULTS_DIR):
        if "test_result.csv" in files:
            test_csv = os.path.join(root, "test_result.csv")
            break
    if test_csv is None:
        print(f"ERROR: test_result.csv not found under {RESULTS_DIR}. Run fine-tuning first.")
        exit(1)

    pred_df = pd.read_csv(test_csv, index_col=0, header=[0, 1])
    probs = pred_df[("preds", TASK_NAME)].values
    targets = pred_df[("targets", TASK_NAME)].values.astype(int)
    smiles = pred_df.index.values

    # Load original test data for strain+S9 info
    orig_df = pd.read_csv(os.path.join(DATA_DIR, "test_ood_fixed_df.csv"))

    # Align by SMILES (test_result.csv index = smiles)
    orig_df = orig_df.set_index("SMILES")
    orig_df = orig_df.loc[smiles]

    # Overall metrics
    overall = compute_metrics(targets, probs)
    print(f"OVERALL ({overall['N_test']} samples):")
    print(f"  AUROC={overall['AUROC']:.4f}  BA={overall['Balanced_Acc']:.4f}  "
          f"Sens={overall['Sensitivity']:.4f}  Spec={overall['Specificity']:.4f}  "
          f"F1={overall['F1']:.4f}  MCC={overall['MCC']:.4f}")
    print()

    # Per strain+S9 breakdown
    rows = []
    orig_df["_probs"] = probs
    orig_df["_targets"] = targets

    for strain in sorted(orig_df["Strain"].unique()):
        for s9 in [0, 1]:
            mask = (orig_df["Strain"] == strain) & (orig_df["S9"] == s9)
            subset = orig_df[mask]
            if len(subset) == 0:
                continue

            s9_label = "with_S9" if s9 == 1 else "without_S9"
            variant = f"{strain}_{s9_label}"
            m = compute_metrics(subset["_targets"].values, subset["_probs"].values)
            m["Variant"] = variant
            rows.append(m)

            print(f"  {variant:25s}: N={m['N_test']:4d}  AUROC={m['AUROC']:.4f}  "
                  f"BA={m['Balanced_Acc']:.4f}  Sens={m['Sensitivity']:.4f}  "
                  f"Spec={m['Specificity']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")

    # Save summary
    summary_df = pd.DataFrame(rows)
    cols = ["Variant", "N_test", "N_pos", "Pos_rate", "AUROC", "Balanced_Acc",
            "Sensitivity", "Specificity", "F1", "MCC"]
    summary_df = summary_df[cols]

    # Add overall row
    overall["Variant"] = "OVERALL"
    summary_df = pd.concat([summary_df, pd.DataFrame([overall])[cols]], ignore_index=True)

    out_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"Summary across {len(rows)} variants:")
    valid = summary_df[summary_df["Variant"] != "OVERALL"]
    for metric in ["AUROC", "Balanced_Acc", "Sensitivity", "Specificity", "F1", "MCC"]:
        vals = valid[metric].dropna().values
        print(f"  {metric:15s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    print(f"\nSaved to: {out_path}")
