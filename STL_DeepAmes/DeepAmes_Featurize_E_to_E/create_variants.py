import pandas as pd
import os

name_of_files = "Train_Val"

# Load the source data
source_file = "/home/ubuntu/Desktop/DeepAmes_Featurize_E_to_E/Fixed_Leakage_Master_Data/train_val_ood_master_df.csv"
df = pd.read_csv(source_file)

# Create output directory
output_dir = f"/home/ubuntu/Desktop/DeepAmes_Featurize_E_to_E/STL_Data/Dataset_Variants_{name_of_files}"
os.makedirs(output_dir, exist_ok=True)

# Define the strains and S9 conditions
strains = ["TA100", "TA102", "TA104", "TA1535", "TA1537", "TA1538", "TA97", "TA98"]
s9_conditions = [(0, "without_S9"), (1, "with_S9")]

# Create 16 variants
for strain in strains:
    for s9_value, s9_label in s9_conditions:
        # Filter the dataframe
        filtered_df = df[(df["Strain"] == strain) & (df["S9"] == s9_value)]
        
        # Create filename
        filename = f"{strain}_{s9_label}_{name_of_files}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        filtered_df.to_csv(filepath, index=False)
        
        print(f"Created: {filename} ({len(filtered_df)} rows)")

print(f"\nAll 16 variants saved to: {output_dir}")
