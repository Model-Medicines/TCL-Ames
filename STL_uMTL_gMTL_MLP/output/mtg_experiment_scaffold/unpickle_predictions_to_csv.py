"""
Convert test_predictions .pkl files to CSV in test_predictions_csv/{model_name}/.
Model name (STL, uMTL, gMTL) is inferred from task count, matching gmtames/results.py.
Each pickle has: gmtamesqsar_id, y_true, y_pred (task columns).
For gMTL, one row per (sample, task) is kept by best-performing grouping (same as results.py).
"""
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report


def task_list_from_filename(fname: str) -> list[str]:
    """Same task_list parsing as gmtames/results.py loadTestPredictions."""
    raw = fname.replace('_test_predictions.pkl', '')
    task_list = raw.split('_TA')
    for i, task in enumerate(task_list):
        if not task.startswith('TA'):
            task_list[i] = 'TA' + task
    return [t for t in task_list if len(t) > 0]


def model_name_from_task_count(num_tasks: int) -> str:
    """STL=1 task, uMTL=16 tasks, gMTL=2–15 tasks (matches gmtames/results.py)."""
    if num_tasks == 1:
        return 'STL'
    if num_tasks == 16:
        return 'uMTL'
    return 'gMTL'


def task_display(internal: str) -> str:
    """Format task for output: TA97_S9 -> TA97_with_S9, TA97 -> TA97_without_S9."""
    if internal.endswith('_S9'):
        return internal[:-3] + '_with_S9'
    return internal + '_without_S9'


def wide_to_long(df_wide: pd.DataFrame, task_list: list[str], grouping_id: str | None = None) -> pd.DataFrame:
    """Convert wide-format predictions to long format: Task, Ground Truth, Binary Prediction. Drops rows with Ground Truth == -2 (missing). Optionally add grouping_id for gMTL. Task column uses display form (e.g. TA97_with_S9, TA97_without_S9)."""
    long_parts = []
    for task in task_list:
        part = df_wide[['gmtamesqsar_id', f'{task}_y_true', f'{task}_y_pred']].copy()
        part = part.rename(columns={f'{task}_y_true': 'Ground Truth', f'{task}_y_pred': 'y_pred'})
        part['Task'] = task_display(task)
        part['Binary Prediction'] = np.rint(part['y_pred']).astype(int)
        part = part[['gmtamesqsar_id', 'Task', 'Ground Truth', 'Binary Prediction']]
        part = part[part['Ground Truth'] != -2]
        if grouping_id is not None:
            part['grouping'] = grouping_id
        long_parts.append(part)
    return pd.concat(long_parts, ignore_index=True)


def _sensitivity(y_true, y_pred):
    """Recall for class 1 (same as results.py)."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return rep.get('1.0', {}).get('recall', 0.0)


def _specificity(y_true, y_pred):
    """Recall for class 0 (same as results.py)."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return rep.get('0.0', {}).get('recall', 0.0)


def gmtl_keep_best_grouping_per_task(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For each task, keep only rows from the best-performing gMTL grouping (same as results.py).
    Best = highest Balanced accuracy, ROC AUC, Sensitivity, Specificity (full_results sort order).
    """
    metrics_list = []
    for (task, grouping), grp in combined.groupby(['Task', 'grouping']):
        y_true = grp['Ground Truth'].values
        y_pred_bin = grp['Binary Prediction'].values
        bal_acc = balanced_accuracy_score(y_true, y_pred_bin)
        try:
            roc = roc_auc_score(y_true, y_pred_bin)
        except ValueError:
            roc = 0.0
        sens = _sensitivity(y_true, y_pred_bin)
        spec = _specificity(y_true, y_pred_bin)
        metrics_list.append({'Task': task, 'grouping': grouping, 'bal_acc': bal_acc, 'roc_auc': roc, 'sensitivity': sens, 'specificity': spec})
    metrics_df = pd.DataFrame(metrics_list)
    # Same sort order as results.py: Balanced accuracy, ROC AUC, Sensitivity, Specificity (all desc)
    metrics_df = metrics_df.sort_values(
        ['bal_acc', 'roc_auc', 'sensitivity', 'specificity'], ascending=[False, False, False, False], kind='stable'
    )
    best_per_task = metrics_df.drop_duplicates(subset='Task', keep='first')[['Task', 'grouping']]
    # Keep only rows where (Task, grouping) is in best_per_task
    combined = combined.merge(best_per_task, on=['Task', 'grouping'], how='inner')
    combined = combined.drop(columns=['grouping'])
    return combined


def pkl_to_csv(pkl_path: Path, out_dir: Path, long_per_model: dict[str, list[pd.DataFrame]]) -> None:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    fname = pkl_path.name
    task_list = task_list_from_filename(fname)
    model_name = model_name_from_task_count(len(task_list))
    sub_dir = out_dir / model_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    ids = data['gmtamesqsar_id']
    y_true = data['y_true']
    y_pred = data['y_pred']

    # Build wide-format DataFrame: gmtamesqsar_id, then for each task {task}_y_true, {task}_y_pred
    rows = []
    for j, gid in enumerate(ids):
        row = {'gmtamesqsar_id': gid}
        for k, task in enumerate(task_list):
            row[f'{task}_y_true'] = y_true[j, k] if y_true.ndim > 1 else y_true[j]
            row[f'{task}_y_pred'] = y_pred[j, k] if y_pred.ndim > 1 else y_pred[j]
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = sub_dir / fname.replace('.pkl', '.csv')
    df.to_csv(out_path, index=False)
    print(f"{model_name}/{out_path.name}")

    # Accumulate long-format rows for single summary CSV per model (gMTL: tag rows with grouping for best-grouping selection)
    grouping_id = fname.replace('.pkl', '') if model_name == 'gMTL' else None
    long_per_model[model_name].append(wide_to_long(df, task_list, grouping_id=grouping_id))


def main():
    base = Path(__file__).resolve().parent
    pkl_dir = base / 'test_predictions'
    out_dir = base / 'test_predictions_csv'
    out_dir.mkdir(exist_ok=True)

    long_per_model = {'STL': [], 'uMTL': [], 'gMTL': []}
    for pkl_path in sorted(pkl_dir.glob('*_test_predictions.pkl')):
        pkl_to_csv(pkl_path, out_dir, long_per_model)

    # One summary CSV per model: Task, Ground Truth, Binary Prediction (one row per (sample, task))
    for model_name, dfs in long_per_model.items():
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        if model_name == 'gMTL':
            # Same as results.py: keep one row per (sample, task) from the best-performing grouping for that task (Balanced accuracy, then ROC AUC)
            combined = gmtl_keep_best_grouping_per_task(combined)
        else:
            # STL/uMTL: no grouping overlap; drop_duplicates only needed if any (unexpected)
            combined = combined.drop_duplicates(subset=['gmtamesqsar_id', 'Task'], keep='first')
            if 'grouping' in combined.columns:
                combined = combined.drop(columns=['grouping'])
        summary_path = out_dir / model_name / 'predictions.csv'
        combined.to_csv(summary_path, index=False)
        print(f"{model_name}/predictions.csv ({len(combined)} rows)")


if __name__ == '__main__':
    main()
