#!/usr/bin/env python
"""
Generate MOLD2 descriptors from SMILES data using multiprocessing
Processes training and test datasets in parallel for maximum performance
"""

from Mold2_pywrapper import Mold2
from rdkit import Chem
import pandas as pd
import time
import warnings
import multiprocessing as mp
import os
warnings.filterwarnings('ignore')


def process_chunk(args):
    """
    Worker function to process a chunk of molecules in parallel

    Args:
        args: Tuple of (chunk_id, mol_chunk, indices_chunk)

    Returns:
        Tuple of (chunk_id, descriptors_df, indices_chunk, error_message)
    """
    chunk_id, mol_chunk, indices_chunk = args

    # Initialize Mold2 calculator in each worker process
    mold2_calc = Mold2()

    try:
        # Calculate descriptors for this chunk
        descriptors_df = mold2_calc.calculate(mol_chunk)
        return (chunk_id, descriptors_df, indices_chunk, None)
    except Exception as e:
        return (chunk_id, None, indices_chunk, str(e))


def generate_mold2_descriptors(input_csv, output_csv, n_cpus=None):
    """
    Generate MOLD2 descriptors from SMILES data using multiprocessing

    Parameters:
    -----------
    input_csv : str
        Path to input CSV file containing SMILES
    output_csv : str
        Path to save output CSV file with MOLD2 descriptors
    n_cpus : int, optional
        Number of CPUs to use. If None, uses all available CPUs.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data and MOLD2 descriptors
    """

    # Get number of CPUs
    if n_cpus is None:
        n_cpus = os.cpu_count()

    print("="*70)
    print(f"Processing: {os.path.basename(input_csv)}")
    print(f"Using multiprocessing with {n_cpus} CPUs")
    print("="*70)

    # Start overall timer
    start_time_total = time.time()

    # Load data
    print(f"\n1. Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"   - Total molecules: {len(df)}")
    print(f"   ✓ Data loaded")

    # Convert SMILES to RDKit molecules
    print("\n2. Converting SMILES to RDKit molecules...")
    start_time = time.time()

    mol_list = []
    valid_indices = []
    failed_count = 0

    for i, smi in enumerate(df['SMILES'].tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_list.append(mol)
            valid_indices.append(i)
        else:
            print(f"   Warning: Could not parse SMILES at row {i}: {smi}")
            failed_count += 1

    elapsed = time.time() - start_time
    print(f"   - Valid molecules: {len(mol_list)}/{len(df)}")
    if failed_count > 0:
        print(f"   - Failed to parse: {failed_count}")
    print(f"   ✓ Conversion completed in {elapsed:.2f} seconds")

    if len(mol_list) == 0:
        raise ValueError("No valid molecules to process")

    # Prepare chunks for parallel processing
    print(f"\n3. Preparing chunks for parallel processing...")
    chunk_size = max(1, len(mol_list) // n_cpus)
    chunks = []

    for i in range(n_cpus):
        start_idx = i * chunk_size
        if i == n_cpus - 1:
            # Last chunk gets all remaining molecules
            end_idx = len(mol_list)
        else:
            end_idx = start_idx + chunk_size

        if start_idx < len(mol_list):
            mol_chunk = mol_list[start_idx:end_idx]
            indices_chunk = valid_indices[start_idx:end_idx]
            chunks.append((i, mol_chunk, indices_chunk))

    print(f"   - Created {len(chunks)} chunks")
    print(f"   - Molecules per chunk: {[len(c[1]) for c in chunks]}")
    print(f"   ✓ Chunks prepared")

    # Calculate descriptors in parallel
    print(f"\n4. Calculating MOLD2 descriptors using {n_cpus} parallel processes...")
    print("   (This may take a few minutes...)")
    start_time = time.time()

    # Use multiprocessing pool
    with mp.Pool(processes=n_cpus) as pool:
        results = pool.map(process_chunk, chunks)

    elapsed = time.time() - start_time

    # Check for errors
    errors = [(r[0], r[3]) for r in results if r[3] is not None]
    if errors:
        print(f"   ✗ Errors in {len(errors)} chunks:")
        for chunk_id, error in errors:
            print(f"      Chunk {chunk_id}: {error}")
        raise Exception(f"Failed to process {len(errors)} chunks")

    # Combine results from all chunks
    print(f"   ✓ Parallel processing completed in {elapsed:.2f} seconds")
    print(f"   - Combining results from {len(results)} chunks...")

    all_descriptors = []
    all_indices = []

    for chunk_id, desc_df, indices, _ in sorted(results, key=lambda x: x[0]):
        all_descriptors.append(desc_df)
        all_indices.extend(indices)

    # Concatenate all descriptor dataframes
    descriptors_df = pd.concat(all_descriptors, ignore_index=True)

    print(f"   ✓ Results combined successfully!")
    print(f"   - Processing time: {elapsed:.2f} seconds")
    print(f"   - Average per molecule: {elapsed/len(mol_list):.3f} seconds")
    print(f"   - Descriptor shape: {descriptors_df.shape}")
    print(f"   - Number of descriptors: {descriptors_df.shape[1]}")

    # Combine with original data
    print("\n5. Creating final dataframe...")
    result_df = pd.DataFrame()
    result_df['gmtamesQSAR_ID'] = [df['gmtamesQSAR_ID'].iloc[i] for i in all_indices]
    result_df['SMILES'] = [df['SMILES'].iloc[i] for i in all_indices]
    
    # The DeepAmes pipeline expects the label column to be named 'label'
    # Convert Endpoint values: 'Positive' -> 1, 'Negative' -> 0
    endpoint_values = [df['Endpoint'].iloc[i] for i in all_indices]
    result_df['label'] = [1 if v == 'Positive' else 0 for v in endpoint_values]

    for col in descriptors_df.columns:
        result_df[col] = descriptors_df[col].values

    print(f"   - Final shape: {result_df.shape}")
    print(f"   ✓ Dataframe created")

    # Save output
    print(f"\n6. Saving results to {output_csv}...")
    result_df.to_csv(output_csv, index=False)
    print(f"   ✓ Results saved")

    total_elapsed = time.time() - start_time_total
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"Total processing time: {total_elapsed:.2f} seconds")
    print(f"Descriptor calculation time: {elapsed:.2f} seconds")
    print(f"Output file: {output_csv}")
    print(f"Processed: {len(mol_list)} molecules using {n_cpus} CPUs")
    print(f"Final dataset: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
    print("="*70)

    return result_df


if __name__ == "__main__":
    import sys
    import glob

    # Get number of CPUs
    n_cpus = os.cpu_count()

    # Define input and output directories
    input_dir = "/home/ubuntu/Desktop/DeepAmes_Featurize/DeepAmes_STL_Data/Dataset_Variants_Train"
    output_dir = "/home/ubuntu/Desktop/DeepAmes_Featurize/Train_Data_Featurized"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get all CSV files from input directory
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    if len(input_files) == 0:
        print(f"\n✗ ERROR: No CSV files found in {input_dir}")
        sys.exit(1)

    print("\n" + "="*70)
    print("BATCH PROCESSING MODE")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(input_files)}")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    print("="*70 + "\n")

    # Track results
    successful = []
    failed = []

    # Process each file
    for i, input_file in enumerate(input_files, 1):
        filename = os.path.basename(input_file)
        # Create output filename with _mold2 suffix
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}_mold2.csv")

        print(f"\n{'#'*70}")
        print(f"# FILE {i}/{len(input_files)}: {filename}")
        print(f"{'#'*70}")

        try:
            # Process the dataset
            result_df = generate_mold2_descriptors(input_file, output_file, n_cpus)
            successful.append((filename, result_df.shape))

            print(f"\n✓ Successfully processed: {filename}")
            print(f"  Output: {output_file}")
            print(f"  Shape: {result_df.shape}")

        except Exception as e:
            print(f"\n✗ ERROR processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((filename, str(e)))

    # Print final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nSuccessfully processed: {len(successful)}/{len(input_files)} files")
    if successful:
        print("\nSuccessful files:")
        for filename, shape in successful:
            print(f"  ✓ {filename} -> {shape[0]} rows × {shape[1]} columns")

    if failed:
        print(f"\nFailed files: {len(failed)}")
        for filename, error in failed:
            print(f"  ✗ {filename}: {error}")

    print(f"\nOutput directory: {output_dir}")
    print("="*70 + "\n")

    if failed:
        sys.exit(1)
