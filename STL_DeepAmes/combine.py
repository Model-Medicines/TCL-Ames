from pathlib import Path
import pandas as pd

# Base directory containing all results
base_dir = Path("/Users/tylerumansky/Desktop/Recovering SMILES/FDA (DeepAmes STL)/All_Results")
# Directory containing the test data with SMILES
test_data_dir = Path("/Users/tylerumansky/Desktop/Recovering SMILES/FDA (DeepAmes STL)/Ready_Data/Test_Data_Featurized")

# Cache for SMILES lookups so we only read each test file once
smiles_cache = {}

def get_smiles_for_task(task_name):
    """Load SMILES from the corresponding test data file for a given task."""
    if task_name not in smiles_cache:
        test_file = test_data_dir / f"{task_name}_Test_mold2.csv"
        if test_file.exists():
            test_df = pd.read_csv(test_file, usecols=['SMILES'])
            smiles_cache[task_name] = test_df['SMILES'].values
        else:
            print(f"Warning: Test file not found for task {task_name}: {test_file}")
            smiles_cache[task_name] = None
    return smiles_cache[task_name]

# Dictionary to store parent dir name -> list of CSV file paths
results = {}

# Iterate through all subdirectories in All_Results
for parent_dir in base_dir.iterdir():
    if parent_dir.is_dir() and not parent_dir.name.startswith('.'):
        # Get only the test_class prediction CSVs (not performance metrics or base results)
        test_class_dir = parent_dir / "result" / "test_class"
        csv_files = [str(csv_file) for csv_file in test_class_dir.glob("*.csv")] if test_class_dir.exists() else []

        # Add to results dict with parent dir name as key
        if csv_files:  # Only add if there are CSV files
            results[parent_dir.name] = csv_files


all_test_weights_list = []
for i in range(13):
    curr_num = i+6
    curr_test_weight = f"weight{curr_num}"
    all_test_weights_list.append(curr_test_weight)

df_dict = {key: [] for key in all_test_weights_list}


# Process results and recover SMILES from test data
for dir_name, csv_list in results.items():
    curr_task_name = dir_name.split("results_")[1]
    smiles_values = get_smiles_for_task(curr_task_name)
    for curr_csv in csv_list:
        curr_matched_weight = curr_csv.split("_")[-1].split(".")[0]
        curr_weight_num = "".join(c for c in curr_matched_weight if c.isdigit())
        curr_column_name = f"class_weight{curr_weight_num}"
        curr_sub_df = pd.read_csv(curr_csv)
        curr_sub_df['Task'] = [curr_task_name] * len(curr_sub_df)
        # Add SMILES column from the corresponding test data (aligned by row index)
        if smiles_values is not None:
            curr_sub_df['SMILES'] = smiles_values
        else:
            curr_sub_df['SMILES'] = None
        curr_sub_df = curr_sub_df[['Task', 'SMILES', 'y_true', curr_column_name]]
        curr_sub_df = curr_sub_df.rename(columns={'y_true': 'Ground Truth', curr_column_name: 'Binary Prediction'})
        df_dict[curr_matched_weight].append(curr_sub_df)


for curr_key in list(df_dict.keys()):
    list_of_dfs = df_dict[curr_key]
    curr_weight_full_preds = pd.concat(list_of_dfs).reset_index(drop=True)
    curr_df_name = f"{curr_key}_DeepAmes_predictions"
    curr_weight_full_preds.to_csv(f"/Users/tylerumansky/Desktop/Recovering SMILES/FDA (DeepAmes STL)/Final_Results/{curr_df_name}.csv",index=False)
    