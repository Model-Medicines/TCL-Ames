import hashlib
import json
import os

import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    from sklearn.externals import joblib


def _absolute(path):
    return os.path.abspath(path)


def dump_joblib(obj, path):
    output_dir = os.path.dirname(path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(obj, path)
    return _absolute(path)


def persist_base_artifact(artifact_dir, col_name, estimator, scaler, model_family, seed, fold, paras_id, hyperparams):
    estimators_dir = os.path.join(artifact_dir, 'estimators')
    scalers_dir = os.path.join(artifact_dir, 'scalers')

    estimator_path = dump_joblib(estimator, os.path.join(estimators_dir, col_name + '.joblib'))
    scaler_path = dump_joblib(scaler, os.path.join(scalers_dir, col_name + '_scaler.joblib'))

    return {
        'name': col_name,
        'model_family': model_family,
        'seed': int(seed),
        'skf_fold': int(fold),
        'paras_id': str(paras_id),
        'hyperparams': json.dumps(hyperparams, sort_keys=True),
        'estimator_path': estimator_path,
        'scaler_path': scaler_path,
    }


def write_artifact_index(records, artifact_dir, filename='artifact_index.csv'):
    os.makedirs(artifact_dir, exist_ok=True)
    output_path = os.path.join(artifact_dir, filename)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return _absolute(output_path)


def sha256_for_columns(columns):
    payload = '||'.join(columns)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def write_json(data, path):
    output_dir = os.path.dirname(path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
    return _absolute(path)
