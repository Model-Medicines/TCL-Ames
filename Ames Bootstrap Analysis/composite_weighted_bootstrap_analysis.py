"""
Bootstrap Statistical Analysis
Purpose: Compute sample-size weighted task-averaged metrics with bootstrap confidence intervals
Method: Bootstrap of a weighted composite estimator

Steps:
1) Compute observed metric per task, then compute a sample-size weighted average across tasks (point estimate)
2) For x in 1..X:
     - within each task, draw one stratified bootstrap resample
     - compute metric per task
     - take the SAME sample-size weighted average across tasks -> one weighted metric for this replicate
3) CI = percentiles of the bootstrap distribution of weighted task-averaged metrics
"""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    recall_score,
    precision_score
)

np.random.seed(42)


def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tn + fp
    return tn / denom if denom != 0 else np.nan


def ppv_score(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def bootstrap_resample_stratified(y_true: pd.Series, y_pred: pd.Series):
    """One stratified bootstrap resample within a task"""

    positives = y_true[y_true == 1]
    negatives = y_true[y_true == 0]

    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0 or n_neg == 0:
        return y_true, y_pred

    pos_idx = positives.iloc[np.random.randint(0, n_pos, size=n_pos)].index
    neg_idx = negatives.iloc[np.random.randint(0, n_neg, size=n_neg)].index

    idx = pos_idx.append(neg_idx)
    return y_true.loc[idx], y_pred.loc[idx]


def bootstrap_task_averaged_ci(task_data_dict, metric_func, n_bootstraps=1000, ci_level=0.95):
    """
    Bootstrap of a sample-size weighted task-averaged estimator.

    Returns:
        avg_observed: weighted point estimate on original data
        avg_lower_ci / avg_upper_ci: CI from bootstrap distribution of weighted task-averaged metrics
        per_task_observed: dataframe of per-task observed values
    """

    task_names = list(task_data_dict.keys())

    # --- sample-size weights (fixed across bootstraps) ---
    ns = np.array([len(task_data_dict[t][0]) for t in task_names], dtype=float)
    weights = ns / ns.sum()

    # Point estimate: weighted average of per-task observed metrics
    per_task = []
    observed_vals = []
    for t in task_names:
        y_true, y_pred = task_data_dict[t]
        v = metric_func(y_true, y_pred)
        per_task.append({"task": t, "n": len(y_true), "weight": float(weights[task_names.index(t)]), "observed": v})
        observed_vals.append(v)

    per_task_df = pd.DataFrame(per_task)
    observed_vals = np.array(observed_vals, dtype=float)
    avg_observed = float(np.sum(weights * observed_vals))

    # Bootstrap distribution of weighted task-averaged metric
    boot_stats = []
    for _ in range(n_bootstraps):
        task_metrics = []
        for t in task_names:
            y_true, y_pred = task_data_dict[t]
            y_true_b, y_pred_b = bootstrap_resample_stratified(y_true, y_pred)
            task_metrics.append(metric_func(y_true_b, y_pred_b))

        task_metrics = np.array(task_metrics, dtype=float)
        if np.any(np.isnan(task_metrics)): print("NaN detected in task metrics:", task_metrics); break
        boot_stats.append(float(np.sum(weights * task_metrics)))

    alpha = 1 - ci_level
    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "avg_observed": float(avg_observed),
        "avg_lower_ci": float(lower),
        "avg_upper_ci": float(upper),
        "bootstrap_estimates": boot_stats,
        "per_task_observed": per_task_df
    }


def load_task_data_from_dataframe(
    df: pd.DataFrame,
    task_col: str | None = "Task",
    true_col: str = "Ground Truth",
    pred_col: str = "Binary Prediction",
    sample_col: str | None = None
) -> dict[str, tuple[pd.Series, pd.Series]]:

    required_cols = [true_col, pred_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    if task_col and task_col in df.columns:
        tasks = list(df[task_col].unique())
    else:
        tasks = ["ALL"]
        task_col = None

    print(f"Found {len(tasks)} task(s): {tasks}\n")

    task_data = {}
    for task in tasks:
        if task_col:
            task_df = df[df[task_col] == task].copy()
        else:
            task_df = df.copy()

        if sample_col and sample_col in df.columns:
            task_df = task_df.sort_values(by=sample_col).reset_index(drop=True)
        else:
            task_df = task_df.reset_index(drop=True)

        y_true = pd.Series(task_df[true_col].values, name=f"{task}_true").astype(int)
        y_pred = pd.Series(task_df[pred_col].values, name=f"{task}_pred").astype(int)
        y_pred.index = y_true.index
        task_data[task] = (y_true, y_pred)
        print(f"  {task}: {len(y_true)} samples")

    print()
    return task_data


def load_from_csv(csv_path, task_col="Task", true_col="Ground Truth", pred_col="Binary Prediction", sample_col=None):

    df = pd.read_csv(csv_path)
    print(f"Loading data from CSV: {csv_path}")
    return load_task_data_from_dataframe(
        df=df,
        task_col=task_col,
        true_col=true_col,
        pred_col=pred_col,
        sample_col=sample_col
    )


def run_analysis(task_data_dict, metric_func, metric_name, n_bootstraps=1000, ci_level=0.95):

    ci_percent = int(round(ci_level * 100))
    print("-" * 60)
    print(f"BOOTSTRAP (WEIGHTED TASK-AVERAGED): {metric_name}")
    print("-" * 60)
    print(f"Number of tasks: {len(task_data_dict)}")
    print(f"Bootstrap replicates: {n_bootstraps}")
    print(f"Confidence level: {ci_percent}%")
    print("-" * 60)

    results = bootstrap_task_averaged_ci(task_data_dict, metric_func, n_bootstraps, ci_level)

    print(f"\nObserved (weighted avg across tasks): {results['avg_observed']:.2f}")
    print(f"{ci_percent}% CI: ({results['avg_lower_ci']:.2f} - {results['avg_upper_ci']:.2f})")

    return results


def save_results_to_csv(all_results, output_path="bootstrap_results.csv", formatted_col_name="Formatted"):
    rows = []
    for metric_name, results in all_results.items():
        rows.append({
            "Metric": metric_name,
            formatted_col_name: f"{results['avg_observed']:.2f} ({results['avg_lower_ci']:.2f}-{results['avg_upper_ci']:.2f})"
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nResults saved to CSV: {output_path}")



def run_all_metrics(task_data_dict=None, csv_path=None, n_bootstraps=1000, ci_level=0.95, save_csv=None, formatted_col_name="Formatted"):

    if csv_path is not None:
        task_data_dict = load_from_csv(csv_path)
    elif task_data_dict is None:
        raise ValueError("Must provide either task_data_dict or csv_path")

    metrics = {
        "Sensitivity": sensitivity_score,
        "BA": balanced_accuracy_score,
        "MCC": matthews_corrcoef,
        "Specificity": specificity_score,
        "PPV": ppv_score,
    }

    all_results = {}
    for metric_name, metric_func in metrics.items():
        print("\n")
        all_results[metric_name] = run_analysis(task_data_dict, metric_func, metric_name, n_bootstraps, ci_level)

    print("\n" + "-" * 80)
    print("SUMMARY: ALL METRICS (WEIGHTED TASK-AVERAGED)")
    print("-" * 80)
    print(f"{'Metric':<15} {'Observed':<12} {'Lower CI':<12} {'Upper CI':<12} {'Formatted':<20}")
    print("-" * 80)

    for metric_name, results in all_results.items():
        obs = results["avg_observed"]
        lower = results["avg_lower_ci"]
        upper = results["avg_upper_ci"]
        formatted = f"{obs:.2f} ({lower:.2f}-{upper:.2f})"
        print(f"{metric_name:<15} {obs:<12.2f} {lower:<12.2f} {upper:<12.2f} {formatted:<20}")

    print("-" * 80)

    if save_csv:
        ci_percent = int(round(ci_level * 100))
        col = f"{formatted_col_name} ({ci_percent}% CI)"
        save_results_to_csv(all_results, save_csv, col)

    return all_results


def get_csv_paths(dir_path: Path | str) -> list[Path]:
    p = Path(dir_path)
    if not p.is_dir():
        return []
    return sorted(p.glob("*.csv"))


def is_wide_prediction_file(csv_path: Path | str) -> bool:
    cols = set(pd.read_csv(csv_path, nrows=0).columns)
    has_endpoint = "Endpoint" in cols
    has_wide_predictions = any(c.endswith("_Binary_Prediction") for c in cols)
    has_long_prediction_col = "Binary Prediction" in cols
    return has_endpoint and has_wide_predictions and not has_long_prediction_col


def run_wide_prediction_file_analysis(
    csv_path: Path | str,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95
) -> dict[str, dict]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loading wide-format data from CSV: {csv_path}")

    prediction_cols = sorted([c for c in df.columns if c.endswith("_Binary_Prediction")])
    if not prediction_cols:
        raise ValueError(f"No *_Binary_Prediction columns found in {csv_path}")

    all_models_results = {}
    for pred_col in prediction_cols:
        model_name = pred_col.replace("_Binary_Prediction", "")
        print(f"\nRunning model from wide file column: {pred_col}")
        task_data = load_task_data_from_dataframe(
            df=df,
            task_col=None,
            true_col="Endpoint",
            pred_col=pred_col
        )
        all_models_results[model_name] = run_all_metrics(
            task_data_dict=task_data,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            save_csv=None,
            formatted_col_name=model_name
        )
        print(f"\nAnalysis completed for model: {model_name}")

    return all_models_results


def run_directory_analysis(
    input_dir: Path | str,
    output_csv_path: Path | str,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95
) -> pd.DataFrame | None:
    csv_paths = get_csv_paths(input_dir)

    if not csv_paths:
        print(f"No CSV files found in {input_dir}")
        return None

    all_models_results = {}
    for csv_path in csv_paths:
        if is_wide_prediction_file(csv_path):
            wide_results = run_wide_prediction_file_analysis(
                csv_path=csv_path,
                n_bootstraps=n_bootstraps,
                ci_level=ci_level
            )
            all_models_results.update(wide_results)
            continue

        model_name = csv_path.stem
        all_models_results[model_name] = run_all_metrics(
            csv_path=str(csv_path),
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            save_csv=None,
            formatted_col_name=model_name
        )
        print(f"\nAnalysis completed for model: {model_name}")

    ci_percent = int(round(ci_level * 100))
    rows = []
    for metric_name in next(iter(all_models_results.values())).keys():
        row = {"Metric": metric_name}
        for model_name, results in all_models_results.items():
            r = results[metric_name]
            row[f"{model_name} ({ci_percent}% CI)"] = f"{r['avg_observed']:.2f} ({r['avg_lower_ci']:.2f}-{r['avg_upper_ci']:.2f})"
        rows.append(row)

    merged = pd.DataFrame(rows)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv_path, index=False)
    print(f"\nSaved results to {output_csv_path}")
    return merged



if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    raw_data_dir = script_dir / "Raw_Prediction_Data"
    output_dir = script_dir / "Bootstrapped_Data_95_CI"

    datasets = {
        "foil_test_preds_bootstrap_results.csv": raw_data_dir / "Foil_Test_Preds",
        "lui_test_preds_bootstrap_results.csv": raw_data_dir / "Lui_Test_Preds",
    }

    for output_file_name, input_dir in datasets.items():
        print("\n" + "=" * 80)
        print(f"Running bootstrap analysis for: {input_dir}")
        print("=" * 80)
        run_directory_analysis(
            input_dir=input_dir,
            output_csv_path=output_dir / output_file_name,
            n_bootstraps=1000,
            ci_level=0.95
        )