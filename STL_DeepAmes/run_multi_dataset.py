#!/usr/bin/env python
"""
Run DeepAmes pipeline across multiple strain/S9 datasets.

This script automatically discovers all train/test dataset pairs in the
Ready_Data directory and runs the full DeepAmes pipeline for each one,
saving outputs in separate directories with clean dataset-specific names.

Usage:
    python run_multi_dataset.py
"""

import os
import glob
import time
from main import run_pipeline
import generate_metrics_report

def get_dataset_pairs(train_dir, test_dir):
    """
    Find matching train/test dataset pairs based on naming convention.
    
    Expects files named: {STRAIN}_{S9_condition}_Train_Val_mold2.csv
    and matching:        {STRAIN}_{S9_condition}_Test_mold2.csv
    
    Parameters:
    -----------
    train_dir : str
        Path to directory containing training CSV files
    test_dir : str
        Path to directory containing test CSV files
        
    Returns:
    --------
    list of tuples: (train_path, test_path, dataset_name)
    """
    train_files = glob.glob(os.path.join(train_dir, "*_Train_Val_mold2.csv"))
    
    pairs = []
    for train_path in sorted(train_files):
        # Extract dataset identifier (e.g., "TA104_with_S9")
        filename = os.path.basename(train_path)
        dataset_name = filename.replace("_Train_Val_mold2.csv", "")
        
        # Find matching test file
        test_filename = f"{dataset_name}_Test_mold2.csv"
        test_path = os.path.join(test_dir, test_filename)
        
        if os.path.exists(test_path):
            pairs.append((train_path, test_path, dataset_name))
        else:
            print(f"Warning: No matching test file for {dataset_name}")
    
    return pairs


def run_all_datasets(base_dir='./Ready_Data', output_base='./results'):
    """
    Run the DeepAmes pipeline for all dataset pairs found in base_dir.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing Train_Data_Featurized and Test_Data_Featurized subdirs
    output_base : str
        Base path for output directories (will append dataset name)
        
    Returns:
    --------
    dict: Results dictionary with status and timing for each dataset
    """
    train_dir = os.path.join(base_dir, 'Train_Data_Featurized')
    test_dir = os.path.join(base_dir, 'Test_Data_Featurized')
    features_path = os.path.join(base_dir, 'all_features.csv')
    
    # Validate directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test data directory not found: {test_dir}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    # Find all dataset pairs
    dataset_pairs = get_dataset_pairs(train_dir, test_dir)
    
    if not dataset_pairs:
        print("No dataset pairs found!")
        return {}
    
    print("=" * 80)
    print("DeepAmes Multi-Dataset Pipeline")
    print("=" * 80)
    print(f"\nFound {len(dataset_pairs)} dataset pair(s):")
    for train, test, name in dataset_pairs:
        print(f"  - {name}")
    print()
    
    results = {}
    total_start = time.time()
    
    for i, (train_path, test_path, dataset_name) in enumerate(dataset_pairs, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(dataset_pairs)}] Processing: {dataset_name}")
        print("=" * 80)
        print(f"  Train: {train_path}")
        print(f"  Test:  {test_path}")
        
        # Create unique output directory for this dataset
        output_dir = f'{output_base}_{dataset_name}'
        
        try:
            # Run the main pipeline
            elapsed = run_pipeline(
                train_data_path=train_path,
                test_data_path=test_path,
                features_path=features_path,
                output_dir=output_dir,
                name=dataset_name
            )
            
            # Generate metrics report for this dataset
            print(f"\nGenerating metrics report for {dataset_name}...")
            report_file = os.path.join(output_dir, f'metrics_report_{dataset_name}.txt')
            generate_metrics_report.generate_report(
                result_dir=output_dir,
                output_file=report_file,
                dataset_name=dataset_name
            )
            
            results[dataset_name] = {
                'status': 'success', 
                'time': elapsed,
                'output_dir': output_dir,
                'report': report_file
            }
            print(f"\nCompleted {dataset_name} in {elapsed:.2f} seconds")
            
        except Exception as e:
            results[dataset_name] = {
                'status': 'failed', 
                'error': str(e),
                'output_dir': output_dir
            }
            print(f"\nFailed {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = sum(1 for r in results.values() if r['status'] == 'failed')
    
    print(f"\nTotal datasets: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    
    print("\nDetailed Results:")
    print("-" * 80)
    for name, result in results.items():
        if result['status'] == 'success':
            print(f"  ✓ {name}")
            print(f"      Time: {result['time']:.2f}s")
            print(f"      Output: {result['output_dir']}")
            print(f"      Report: {result['report']}")
        else:
            print(f"  ✗ {name}")
            print(f"      Error: {result['error']}")
    
    return results


def run_single_dataset(dataset_name, base_dir='./Ready_Data', output_base='./results'):
    """
    Run the pipeline for a single specified dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., "TA104_with_S9")
    base_dir : str
        Base directory containing the data
    output_base : str
        Base path for output directory
        
    Returns:
    --------
    dict: Result dictionary with status and timing
    """
    train_dir = os.path.join(base_dir, 'Train_Data_Featurized')
    test_dir = os.path.join(base_dir, 'Test_Data_Featurized')
    features_path = os.path.join(base_dir, 'all_features.csv')
    
    train_path = os.path.join(train_dir, f"{dataset_name}_Train_Val_mold2.csv")
    test_path = os.path.join(test_dir, f"{dataset_name}_Test_mold2.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    output_dir = f'{output_base}_{dataset_name}'
    
    print("=" * 80)
    print(f"Processing: {dataset_name}")
    print("=" * 80)
    
    elapsed = run_pipeline(
        train_data_path=train_path,
        test_data_path=test_path,
        features_path=features_path,
        output_dir=output_dir,
        name=dataset_name
    )
    
    # Generate metrics report
    print(f"\nGenerating metrics report for {dataset_name}...")
    report_file = os.path.join(output_dir, f'metrics_report_{dataset_name}.txt')
    generate_metrics_report.generate_report(
        result_dir=output_dir,
        output_file=report_file,
        dataset_name=dataset_name
    )
    
    print(f"\nCompleted {dataset_name} in {elapsed:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"Metrics report: {report_file}")
    
    return {
        'status': 'success',
        'time': elapsed,
        'output_dir': output_dir,
        'report': report_file
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run DeepAmes pipeline on multiple datasets'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help='Specific dataset to run (e.g., "TA104_with_S9"). If not provided, runs all datasets.'
    )
    parser.add_argument(
        '--base-dir', '-b',
        type=str,
        default='./Ready_Data',
        help='Base directory containing Train_Data_Featurized and Test_Data_Featurized (default: ./Ready_Data)'
    )
    parser.add_argument(
        '--output-base', '-o',
        type=str,
        default='./results',
        help='Base path for output directories (default: ./results)'
    )
    
    args = parser.parse_args()
    
    if args.dataset:
        # Run single dataset
        run_single_dataset(
            dataset_name=args.dataset,
            base_dir=args.base_dir,
            output_base=args.output_base
        )
    else:
        # Run all datasets
        run_all_datasets(
            base_dir=args.base_dir,
            output_base=args.output_base
        )
