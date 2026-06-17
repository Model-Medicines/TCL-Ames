from pathlib import Path
import re

import pandas as pd

# Resolve project-relative paths from this script location
project_root = Path(__file__).resolve().parent

# Base directory containing inference result CSVs
base_dir = project_root / "Inference_Results"
# Directory for combined final CSV outputs
final_results_dir = project_root / "Final_Results"
final_results_dir.mkdir(parents=True, exist_ok=True)


# File pattern example: TA98_without_S9_weight6_inference.csv
inference_file_pattern = re.compile(r"(?P<task>.+)_weight(?P<weight>\d+)_inference\.csv$")

# Dictionary to store weight key -> list of task-specific dataframes
df_dict = {}

for csv_file in sorted(base_dir.glob("*.csv")):
    match = inference_file_pattern.match(csv_file.name)
    if not match:
        print(f"Skipping file with unexpected format: {csv_file.name}")
        continue

    task_name = match.group("task")
    weight_num = match.group("weight")
    weight_key = f"weight{weight_num}"
    prob_col = f"prob_weight{weight_num}"
    class_col = f"class_weight{weight_num}"

    curr_sub_df = pd.read_csv(csv_file)

    missing_cols = {"SMILES", "y_true", prob_col, class_col} - set(curr_sub_df.columns)
    if missing_cols:
        print(f"Skipping {csv_file.name}: missing columns {sorted(missing_cols)}")
        continue

    curr_sub_df["Task"] = task_name
    curr_sub_df = curr_sub_df[["Task", "SMILES", "y_true", prob_col, class_col]]
    curr_sub_df = curr_sub_df.rename(
        columns={
            "y_true": "Ground Truth",
            prob_col: "Probability",
            class_col: "Binary Prediction",
        }
    )

    df_dict.setdefault(weight_key, []).append(curr_sub_df)


for curr_key, list_of_dfs in df_dict.items():
    if not list_of_dfs:
        continue

    curr_weight_full_preds = pd.concat(list_of_dfs).reset_index(drop=True)
    curr_df_name = f"{curr_key}_DeepAmes_inference_predictions"
    curr_weight_full_preds.to_csv(final_results_dir / f"{curr_df_name}.csv", index=False)
    print(f"Saved {curr_df_name}.csv with {len(curr_weight_full_preds)} rows")
