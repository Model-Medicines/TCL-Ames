import pandas as pd
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import itertools
from functools import reduce
import os.path

def _read_artifact_index(model, basepath, artifact_base_path):
    candidates = []
    if artifact_base_path is not None:
        candidates.append(join(artifact_base_path, model, 'artifact_index.csv'))
    candidates.append(join(basepath, model, 'artifacts', 'artifact_index.csv'))

    for filepath in candidates:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
    return None

def sep_performance(filepath):
    df = pd.read_csv(filepath)
    df = df.iloc[:, 1:]
    df = df.rename(columns={'0':'name', '1':'value'})
    df['name'] = df['name'].astype('str')
    df['value'] = df['value'].astype('str')
    df['model'] = df['name'].str.split('_').str[0].values
    df['seed'] = df['name'].str.split('_paras_').str[0].str.split('_seed_').str[1]
    
    
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC']
    for i, col in enumerate(cols):
        if i == 0:
            df[col] = df.value.str.split(',').str[i].str.split('[').str[1].values
        elif i == len(cols)-1:
            df[col] = df.value.str.split(',').str[i].str.split(']').str[0].values
        else:
            df[col] = df.value.str.split(',').str[i].values

    for i, col in enumerate(cols):
        if i < 4:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
            df[col] = round(df[col], 3)
    del df['value']
            
    return df

def select_base_classifiers(basepath, artifact_base_path=None, selected_artifact_output_path=None):

    xgboost = sep_performance(basepath + '/xgboost/validation_performance/validation_xgboost_paras_151.csv')
    rf = sep_performance(basepath + '/rf/validation_performance/validation_rf__paras_148.csv') 
    svm = sep_performance(basepath + '/svm/validation_performance/validation_svm_paras_19.csv') 
    lr = sep_performance(basepath + '/lr/validation_performance/validation_lr_paras_22.csv')
    knn = sep_performance(basepath + '/knn/validation_performance/validation_knn_paras_0.csv')

    result = pd.concat([xgboost, rf, svm, lr, knn], axis=0)
    
    subresult = result[(result.MCC <np.percentile(result.MCC.values, 95)) & (result.MCC > np.percentile(result.MCC.values, 5))]
    subresult = subresult.reset_index(drop=True)
    
    subresult.to_csv(basepath + '/selected_base_classifiers.csv')

    artifact_indexes = []
    for model in ['knn', 'lr', 'svm', 'rf', 'xgboost']:
        idx = _read_artifact_index(model, basepath, artifact_base_path)
        if idx is not None and len(idx) > 0:
            artifact_indexes.append(idx)
        else:
            print('Warning: artifact index not found for model:', model)

    if len(artifact_indexes) > 0:
        artifact_df = pd.concat(artifact_indexes, axis=0, ignore_index=True)
        selected_artifacts = subresult.merge(
            artifact_df,
            on='name',
            how='left'
        )
    else:
        selected_artifacts = subresult.copy()
        selected_artifacts['estimator_path'] = np.nan
        selected_artifacts['scaler_path'] = np.nan

    if selected_artifact_output_path is None:
        if artifact_base_path is not None:
            selected_artifact_output_path = join(artifact_base_path, 'selected_base_artifacts.csv')
        else:
            selected_artifact_output_path = basepath + '/selected_base_artifacts.csv'
    output_dir = os.path.dirname(selected_artifact_output_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    selected_artifacts.to_csv(selected_artifact_output_path, index=False)
    return subresult