#!/usr/bin/env python

import time

### Warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

### import scripts
import base_knn
import base_lr
import base_svm
import base_rf
import base_xgboost

import select_base

import validation_predictions_combine
import test_predictions_combine

import deepames_plus

from tqdm import tqdm
import os


def run_pipeline(train_data_path, test_data_path, features_path, output_dir, name):
    """
    Run the full DeepAmes training pipeline.
    
    Parameters:
    -----------
    train_data_path : str
        Path to the training data CSV file
    test_data_path : str
        Path to the test data CSV file
    features_path : str
        Path to the features CSV file
    output_dir : str
        Directory where all outputs will be saved
    name : str
        Name identifier for this run
    """
    start_time = time.time()
    
    ### define the path for data, base classifers, dnn results 
    features = pd.read_csv(features_path).feature.unique()
    data = pd.read_csv(train_data_path, low_memory=False)
    external = pd.read_csv(test_data_path)

    workDir = output_dir
    base_path = workDir + '/base'
    probability_path = workDir + '/probabilities_output'
    result_path = workDir + '/result'
    model_path = workDir + '/DeepAmes_models'

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(probability_path, exist_ok=True)
    os.makedirs(result_path + '/validation_class', exist_ok=True)
    os.makedirs(result_path + '/validation_performance', exist_ok=True)
    os.makedirs(result_path + '/test_class', exist_ok=True)
    os.makedirs(result_path + '/test_performance', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    ### run the scripts
    base_knn.generate_baseClassifiers(features, data, external, base_path+'/knn')
    base_lr.generate_baseClassifiers(features, data, external, base_path+'/lr')
    base_svm.generate_baseClassifiers(features, data, external, base_path+'/svm')
    base_rf.generate_baseClassifiers(features, data, external, base_path+'/rf')
    base_xgboost.generate_baseClassifiers(features, data, external, base_path+'/xgboost')

    mcc = select_base.select_base_classifiers(base_path)

    validation_predictions_combine.combine_validation_probabilities(base_path, mcc, probability_path, name)
    test_predictions_combine.combine_test_probabilities(base_path, mcc, probability_path, name)

    for weight in tqdm(range(6, 19, 1), desc="DeepAmes+ Weights"):
        deepames_plus.deepames_prediction(probability_path, name, weight, result_path, model_path)

    elapsed_time = time.time() - start_time
    print("--- %s seconds ---" % elapsed_time)
    return elapsed_time


# Default behavior when run as a script (backward compatible) - use run_multi_dataset.py instead
# if __name__ == "__main__":
#     start_time = time.time()
#
#     ### define the path for data, base classifers, dnn results
#     features = pd.read_csv('./Ready_Data/all_features.csv').feature.unique()# path for mold2_selected_features_10523.csv
#     data = pd.read_csv('./Ready_Data/Train_Data_Featurized/TA104_with_S9_Train_Val_mold2.csv', low_memory=False)# path for clean_train_10444.csv
#     external =  pd.read_csv('./Ready_Data/Test_Data_Featurized/TA104_with_S9_Test_mold2.csv') # path for ames_test.csv for test set
#     #external['label'] = np.where(external.index < 3000, 0, 1)
#
#     name = 'ames6512' # can be any name
#
#     workDir = './results'
#     base_path = workDir + '/base'# path for base classifiers
#     probability_path = workDir + '/probabilities_output' # path for the combined probabilities (model-level representations)
#     result_path = workDir + '/result' # path for the final deepcarc predictions
#
#     os.makedirs(base_path, exist_ok=True)
#     os.makedirs(probability_path, exist_ok=True)
#     os.makedirs(result_path + '/validation_class', exist_ok=True)
#     os.makedirs(result_path + '/validation_performance', exist_ok=True)
#     os.makedirs(result_path + '/test_class', exist_ok=True)
#     os.makedirs(result_path + '/test_performance', exist_ok=True)
#     os.makedirs('./DeepAmes_models', exist_ok=True)
#
#     ### run the scripts
#     base_knn.generate_baseClassifiers(features, data, external, base_path+'/knn')
#     base_lr.generate_baseClassifiers(features, data, external, base_path+'/lr')
#     base_svm.generate_baseClassifiers(features, data, external, base_path+'/svm')
#     base_rf.generate_baseClassifiers(features, data, external, base_path+'/rf')
#     base_xgboost.generate_baseClassifiers(features, data, external, base_path+'/xgboost')
#
#     mcc = select_base.select_base_classifiers(base_path)
#
#     validation_predictions_combine.combine_validation_probabilities(base_path, mcc, probability_path, name)
#     test_predictions_combine.combine_test_probabilities(base_path, mcc, probability_path, name)
#
#     #for weight in range(6, 19, 1):
#     for weight in tqdm(range(6, 19, 1), desc="DeepAmes+ Weights"):
#         deepames_plus.deepames_prediction(probability_path, name, weight, result_path, './DeepAmes_models')
#
#
#
#     print("--- %s seconds ---" % (time.time() - start_time))
