#!/usr/bin/env python3
"""
Generate metrics report with bootstrap confidence intervals for DeepAmes models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
    matthews_corrcoef
)
from collections import OrderedDict
import os

def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix components."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "True Negative": int(tn),
        "False Positive": int(fp),
        "False Negative": int(fn),
        "True Positive": int(tp)
    }


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate all metrics from true labels and predictions."""
    sensitivity = recall_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    try:
        precision = precision_score(y_true, y_pred)
    except:
        precision = 0
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = OrderedDict([
        ("Sensitivity", sensitivity),
        ("Specificity", specificity),
        ("Precision (PPV)", precision),
        ("NPV", npv),
        ("Accuracy", accuracy),
        ("Balanced Accuracy", balanced_acc),
        ("F1 Score", f1),
        ("MCC", mcc),
    ])
    
    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["AUC"] = auc
        except:
            pass
    
    return metrics


def stratified_bootstrap_metrics(y_true, y_pred, y_prob=None, n_iterations=1000, random_state=42):
    """
    Perform stratified bootstrap to get confidence intervals for metrics.
    Stratifies by ground truth class to maintain class proportions.
    """
    np.random.seed(random_state)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)
    
    # Calculate observed metrics
    observed_metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Get indices for each class
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    # Bootstrap iterations
    bootstrap_metrics = []
    
    for _ in range(n_iterations):
        # Stratified resampling
        pos_resampled = np.random.choice(pos_indices, size=n_pos, replace=True)
        neg_resampled = np.random.choice(neg_indices, size=n_neg, replace=True)
        
        bootstrap_indices = np.concatenate([pos_resampled, neg_resampled])
        
        y_true_boot = y_true[bootstrap_indices]
        y_pred_boot = y_pred[bootstrap_indices]
        y_prob_boot = y_prob[bootstrap_indices] if y_prob is not None else None
        
        metrics_i = calculate_metrics(y_true_boot, y_pred_boot, y_prob_boot)
        bootstrap_metrics.append(metrics_i)
    
    # Convert to DataFrame for easy quantile calculation
    bootstrap_df = pd.DataFrame(bootstrap_metrics)
    
    # Calculate 90% CI (5th and 95th percentiles)
    ci_lower = bootstrap_df.quantile(0.05).to_dict()
    ci_upper = bootstrap_df.quantile(0.95).to_dict()
    
    # Compile results
    results = OrderedDict()
    for metric in observed_metrics.keys():
        results[metric] = {
            "value": observed_metrics[metric],
            "ci_lower": ci_lower[metric],
            "ci_upper": ci_upper[metric]
        }
    
    return results, observed_metrics


def generate_report(result_dir='./results', output_file=None, dataset_name=''):
    """
    Generate metrics report with bootstrap confidence intervals for DeepAmes models.
    
    Parameters:
    -----------
    result_dir : str
        Path to the results directory containing 'result' subdirectory
        (e.g., './results_TA104_with_S9')
    output_file : str
        Path for the output report file. If None, auto-generated based on result_dir
    dataset_name : str
        Name identifier for the report header (e.g., 'TA104_with_S9')
    """
    result_path = os.path.join(result_dir, 'result')
    test_class_path = os.path.join(result_path, 'test_class')
    
    if output_file is None:
        if dataset_name:
            output_file = os.path.join(result_dir, f'metrics_report_{dataset_name}.txt')
        else:
            output_file = os.path.join(result_dir, 'metrics_report_bootstrap.txt')
    
    weights = range(6, 19)
    n_iterations = 1000
    
    print(f"Processing models with {n_iterations} bootstrap iterations...")
    print("=" * 80)
    
    all_results = {}
    
    for weight in weights:
        print(f"Processing weight {weight}...")
        test_file = f"{test_class_path}/test_weight{weight}.csv"
        
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            
            y_true = df['y_true'].values
            y_pred = df[f'class_weight{weight}'].values
            y_prob = df[f'prob_weight{weight}'].values
            
            # Calculate bootstrap metrics
            bootstrap_results, observed_metrics = stratified_bootstrap_metrics(
                y_true, y_pred, y_prob, n_iterations=n_iterations
            )
            
            # Calculate confusion matrix
            cm = calculate_confusion_matrix(y_true, y_pred)
            
            all_results[weight] = {
                'bootstrap': bootstrap_results,
                'confusion_matrix': cm,
                'n_samples': len(df)
            }
            print(f"  Done. {len(df)} samples.")
    
    print("=" * 80)
    print("All models processed. Generating report...\n")
    
    # Get ground truth stats from first weight
    first_weight = list(all_results.keys())[0]
    n_total = all_results[first_weight]['n_samples']
    cm_first = all_results[first_weight]['confusion_matrix']
    n_pos = cm_first['True Positive'] + cm_first['False Negative']
    n_neg = cm_first['True Negative'] + cm_first['False Positive']
    
    # Write report
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("              DeepAmes MODEL PERFORMANCE REPORT\n")
        if dataset_name:
            f.write(f"                    Dataset: {dataset_name}\n")
        f.write("         Binary Classification with Bootstrap 90% CIs\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Bootstrap: 1000 iterations, stratified by class, 90% CI\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("                      TEST SET STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Compounds:          {n_total}\n")
        f.write(f"Positive (Mutagenic):     {n_pos} ({100*n_pos/n_total:.2f}%)\n")
        f.write(f"Negative (Non-mutagenic): {n_neg} ({100*n_neg/n_total:.2f}%)\n\n")
        
        # Summary Tables
        f.write("=" * 80 + "\n")
        f.write("                         SUMMARY TABLES\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by MCC
        sorted_weights = sorted(all_results.items(), key=lambda x: x[1]['bootstrap']['MCC']['value'], reverse=True)
        
        # Confusion Matrix Summary
        f.write("CONFUSION MATRIX SUMMARY (sorted by MCC)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Weight':<10} {'TN':>8} {'FP':>8} {'FN':>8} {'TP':>8}\n")
        f.write("-" * 80 + "\n")
        for weight, results in sorted_weights:
            cm = results['confusion_matrix']
            f.write(f"Weight {weight:<4} {cm['True Negative']:>8} {cm['False Positive']:>8} {cm['False Negative']:>8} {cm['True Positive']:>8}\n")
        f.write("-" * 80 + "\n\n")
        
        # MCC Ranking
        f.write("MCC RANKING (with 90% CIs)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Weight':<10} {'MCC':<30}\n")
        f.write("-" * 80 + "\n")
        for rank, (weight, results) in enumerate(sorted_weights, 1):
            bs = results['bootstrap']
            mcc = f"{bs['MCC']['value']:.4f} ({bs['MCC']['ci_lower']:.4f}, {bs['MCC']['ci_upper']:.4f})"
            f.write(f"{rank:<6} Weight {weight:<4} {mcc:<30}\n")
        f.write("-" * 80 + "\n\n")
        
        # Sensitivity Ranking
        sens_ranking = sorted(all_results.items(), key=lambda x: x[1]['bootstrap']['Sensitivity']['value'], reverse=True)
        f.write("SENSITIVITY RANKING (with 90% CIs)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Weight':<10} {'Sensitivity':<30} {'FN':>6}\n")
        f.write("-" * 80 + "\n")
        for rank, (weight, results) in enumerate(sens_ranking, 1):
            bs = results['bootstrap']
            sens = f"{bs['Sensitivity']['value']:.4f} ({bs['Sensitivity']['ci_lower']:.4f}, {bs['Sensitivity']['ci_upper']:.4f})"
            fn = results['confusion_matrix']['False Negative']
            f.write(f"{rank:<6} Weight {weight:<4} {sens:<30} {fn:>6}\n")
        f.write("-" * 80 + "\n\n")
        
        # Performance Metrics Summary
        f.write("PERFORMANCE METRICS SUMMARY (with 90% CIs)\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Weight':<10} {'Sensitivity':<25} {'Specificity':<25} {'F1 Score':<25}\n")
        f.write("-" * 110 + "\n")
        for weight, results in sorted_weights:
            bs = results['bootstrap']
            sens = f"{bs['Sensitivity']['value']:.3f} ({bs['Sensitivity']['ci_lower']:.3f}, {bs['Sensitivity']['ci_upper']:.3f})"
            spec = f"{bs['Specificity']['value']:.3f} ({bs['Specificity']['ci_lower']:.3f}, {bs['Specificity']['ci_upper']:.3f})"
            f1 = f"{bs['F1 Score']['value']:.3f} ({bs['F1 Score']['ci_lower']:.3f}, {bs['F1 Score']['ci_upper']:.3f})"
            f.write(f"Weight {weight:<4} {sens:<25} {spec:<25} {f1:<25}\n")
        f.write("-" * 110 + "\n\n")
        
        # Accuracy Metrics Summary
        f.write("ACCURACY METRICS SUMMARY (with 90% CIs)\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Weight':<10} {'Accuracy':<25} {'Balanced Accuracy':<25} {'AUC':<25}\n")
        f.write("-" * 110 + "\n")
        for weight, results in sorted_weights:
            bs = results['bootstrap']
            acc = f"{bs['Accuracy']['value']:.3f} ({bs['Accuracy']['ci_lower']:.3f}, {bs['Accuracy']['ci_upper']:.3f})"
            bal = f"{bs['Balanced Accuracy']['value']:.3f} ({bs['Balanced Accuracy']['ci_lower']:.3f}, {bs['Balanced Accuracy']['ci_upper']:.3f})"
            auc = f"{bs['AUC']['value']:.3f} ({bs['AUC']['ci_lower']:.3f}, {bs['AUC']['ci_upper']:.3f})" if 'AUC' in bs else "N/A"
            f.write(f"Weight {weight:<4} {acc:<25} {bal:<25} {auc:<25}\n")
        f.write("-" * 110 + "\n\n")
        
        # Detailed results for each weight
        f.write("=" * 80 + "\n")
        f.write("                    DETAILED RESULTS BY WEIGHT\n")
        f.write("=" * 80 + "\n")
        
        for weight, results in sorted_weights:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Weight {weight}\n")
            f.write("=" * 80 + "\n\n")
            
            # Confusion Matrix
            cm = results['confusion_matrix']
            f.write("Confusion Matrix:\n")
            f.write("-" * 40 + "\n")
            f.write(f"                         Predicted\n")
            f.write(f"                      Neg (0)    Pos (1)\n")
            f.write(f"    Actual Neg (0)    {cm['True Negative']:5d}      {cm['False Positive']:5d}\n")
            f.write(f"    Actual Pos (1)    {cm['False Negative']:5d}      {cm['True Positive']:5d}\n")
            f.write(f"\n    TN={cm['True Negative']}, FP={cm['False Positive']}, FN={cm['False Negative']}, TP={cm['True Positive']}\n\n")
            
            # Metrics with CIs
            f.write("Metrics with Bootstrap 90% CIs:\n")
            f.write("-" * 60 + "\n")
            
            for metric, values in results['bootstrap'].items():
                f.write(f"{metric}:\n")
                f.write(f"  Value = {values['value']:.4f}\n")
                f.write(f"  90% CI = ({values['ci_lower']:.4f}, {values['ci_upper']:.4f})\n")
        
        # Best model summary
        best_weight, best_results = sorted_weights[0]
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("                           BEST MODEL\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best performing model (by MCC): Weight {best_weight}\n\n")
        bs = best_results['bootstrap']
        f.write(f"MCC:               {bs['MCC']['value']:.4f} ({bs['MCC']['ci_lower']:.4f}, {bs['MCC']['ci_upper']:.4f})\n")
        f.write(f"Sensitivity:       {bs['Sensitivity']['value']:.4f} ({bs['Sensitivity']['ci_lower']:.4f}, {bs['Sensitivity']['ci_upper']:.4f})\n")
        f.write(f"Specificity:       {bs['Specificity']['value']:.4f} ({bs['Specificity']['ci_lower']:.4f}, {bs['Specificity']['ci_upper']:.4f})\n")
        f.write(f"F1 Score:          {bs['F1 Score']['value']:.4f} ({bs['F1 Score']['ci_lower']:.4f}, {bs['F1 Score']['ci_upper']:.4f})\n")
        model_dir = os.path.join(result_dir, 'DeepAmes_models')
        f.write(f"\nModel file: {model_dir}/weight_{best_weight}.h5\n")
        
        # Metric definitions
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("                           METRIC DEFINITIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Sensitivity:      TP / (TP + FN) - Proportion of actual positives correctly identified\n")
        f.write("Specificity:      TN / (TN + FP) - Proportion of actual negatives correctly identified\n")
        f.write("Precision (PPV):  TP / (TP + FP) - Proportion of predicted positives that are correct\n")
        f.write("NPV:              TN / (TN + FN) - Negative Predictive Value\n")
        f.write("Accuracy:         (TP + TN) / Total - Overall correctness\n")
        f.write("Balanced Accuracy: (Sensitivity + Specificity) / 2\n")
        f.write("F1 Score:         2 * (Precision * Recall) / (Precision + Recall)\n")
        f.write("MCC:              Matthews Correlation Coefficient - balanced measure for imbalanced data\n")
        f.write("AUC:              Area Under the ROC Curve\n\n")
        f.write("90% CI:           Bootstrap confidence interval (5th to 95th percentile)\n")
        f.write("                  1000 iterations with stratified resampling\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("                            END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Report saved to: {output_file}")
    print("\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate metrics report with bootstrap confidence intervals'
    )
    parser.add_argument(
        '--result-dir', '-r',
        type=str,
        default='./results',
        help='Path to results directory (default: ./results)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: auto-generated in result_dir)'
    )
    parser.add_argument(
        '--dataset-name', '-n',
        type=str,
        default='',
        help='Dataset name for report header (e.g., "TA104_with_S9")'
    )
    
    args = parser.parse_args()
    
    generate_report(
        result_dir=args.result_dir,
        output_file=args.output,
        dataset_name=args.dataset_name
    )
