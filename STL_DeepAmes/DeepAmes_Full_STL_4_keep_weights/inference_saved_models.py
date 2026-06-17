#!/usr/bin/env python
"""
Inference-only runner for DeepAmes stacked models.

This script does NOT train/retrain any model. It only:
1) Loads saved base-model estimators/scalers.
2) Builds the stacked probability features for the test set (parallel threading + batch reuse).
3) Loads a saved DeepAmes .h5 model and runs forward inference.
"""

import argparse
import json
import os
import re
import sys
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover
    from sklearn.externals import joblib

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None
    delayed = None

# Default parallel workers for stacked-base predict_proba (joblib threading).
# Capped by cpu_count() on smaller hosts.
DEFAULT_BASE_PARALLEL_JOBS = 48


def _require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError("%s not found: %s" % (label, path))
    return path


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _infer_feature_name(row):
    model = str(row.get("model", "")).strip()
    seed_x = str(row.get("seed_x", "")).strip()
    if model != "" and seed_x != "":
        return "%s_seed_%s" % (model, seed_x)

    name = str(row.get("name", "")).strip()
    match = re.match(r"^([A-Za-z0-9]+)_seed_([0-9]+_skf_[0-9]+)_paras_", name)
    if match is None:
        raise ValueError("Unable to infer feature name from row: %s" % name)
    return "%s_seed_%s" % (match.group(1), match.group(2))


def _remap_saved_artifact_path(saved_base_root, dataset_name, model_family, original_path, artifact_kind):
    filename = os.path.basename(str(original_path))
    return os.path.join(
        saved_base_root,
        "results_%s" % dataset_name,
        str(model_family),
        artifact_kind,
        filename,
    )


def _resolve_defaults(args):
    dataset_name = args.dataset_name
    project_root = args.project_root
    base_artifact_index_root = (
        args.base_artifact_index_root
        if args.base_artifact_index_root is not None
        else os.path.join(project_root, "Base_Artifact_Index_CSVs")
    )
    inference_metadata_root = (
        args.inference_metadata_root
        if args.inference_metadata_root is not None
        else os.path.join(project_root, "Inference_Metadata_Artifacts")
    )

    defaults = {
        "features_path": os.path.join(project_root, "Ready_Data", "all_features.csv"),
        "test_data_path": os.path.join(
            project_root,
            "Ready_Data",
            "Test_Data_Featurized",
            "%s_Test_mold2.csv" % dataset_name,
        ),
        "selected_base_csv": os.path.join(
            base_artifact_index_root,
            "results_%s" % dataset_name,
            "artifacts",
            "selected_base_artifacts.csv",
        ),
        "schema_path": os.path.join(
            inference_metadata_root,
            "results_%s" % dataset_name,
            "artifacts",
            "schema",
            "validation_schema_%s.json" % dataset_name,
        ),
        "validation_probabilities_path": os.path.join(
            inference_metadata_root,
            "results_%s" % dataset_name,
            "probabilities_output",
            "validation_probabilities_%s.csv" % dataset_name,
        ),
        "deep_model_path": os.path.join(
            args.saved_deep_root,
            "results_%s" % dataset_name,
            "weight_%s.h5" % args.weight,
        ),
        "deep_scaler_path": os.path.join(
            inference_metadata_root,
            "results_%s" % dataset_name,
            "artifacts",
            "deepames",
            "weight_%s_scaler.joblib" % args.weight,
        ),
        "output_csv": os.path.join(
            project_root,
            "Inference_Results",
            "%s_weight%s_inference.csv" % (dataset_name, args.weight),
        ),
    }

    resolved = {}
    for key, value in defaults.items():
        user_value = getattr(args, key)
        resolved[key] = user_value if user_value is not None else value
    return resolved


def _resolve_base_parallel_jobs(desired_jobs):
    if desired_jobs <= 1:
        return 1
    cap = cpu_count() or desired_jobs
    return max(2, min(int(desired_jobs), cap))


def _effective_base_jobs(cli_raw):
    c = cpu_count() or 16
    if cli_raw is None:
        # Default: use up to DEFAULT_BASE_PARALLEL_JOBS logical CPUs (e.g. 48 of 60).
        return max(1, min(int(DEFAULT_BASE_PARALLEL_JOBS), int(c)))
    if cli_raw <= 1:
        return 1
    return max(2, min(int(cli_raw), int(c)))


def _predict_single_base_probability(row, saved_base_root, dataset_name, x_test):
    model_family = str(row["model_family"])
    estimator_path = _remap_saved_artifact_path(
        saved_base_root,
        dataset_name,
        model_family,
        row["estimator_path"],
        "estimators",
    )
    scaler_path = _remap_saved_artifact_path(
        saved_base_root,
        dataset_name,
        model_family,
        row["scaler_path"],
        "scalers",
    )

    _require_file(estimator_path, "Base estimator")
    _require_file(scaler_path, "Base scaler")

    estimator = joblib.load(estimator_path)
    scaler = joblib.load(scaler_path)
    x_scaled = scaler.transform(x_test)

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x_scaled)
        if len(proba.shape) == 2 and proba.shape[1] > 1:
            pos_prob = proba[:, 1].astype(np.float64, copy=False)
        else:
            pos_prob = np.asarray(proba, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(
            "Estimator does not expose predict_proba: %s" % estimator_path
        )

    feature_name = _infer_feature_name(row)
    return feature_name, pos_prob


def _load_base_probabilities(
    test_df,
    feature_columns,
    selected_base_csv,
    saved_base_root,
    dataset_name,
    parallel_jobs=1,
):
    selected_df = pd.read_csv(selected_base_csv)
    if selected_df.empty:
        raise ValueError("Selected base artifact table is empty: %s" % selected_base_csv)

    missing_features = [col for col in feature_columns if col not in test_df.columns]
    if missing_features:
        raise ValueError(
            "Test data is missing %d required descriptor columns. First few: %s"
            % (len(missing_features), ", ".join(missing_features[:10]))
        )

    x_test = test_df[feature_columns]
    n_rows = len(selected_df)
    want_parallel = (
        parallel_jobs > 1
        and n_rows >= 16
        and Parallel is not None
        and delayed is not None
    )

    if not want_parallel:
        base_probs = {}
        for _, row in selected_df.iterrows():
            feature_name, pos_prob = _predict_single_base_probability(
                row, saved_base_root, dataset_name, x_test
            )
            base_probs[feature_name] = pos_prob
        return pd.DataFrame(base_probs, index=test_df.index)

    pj = _resolve_base_parallel_jobs(parallel_jobs)
    pairs = Parallel(n_jobs=pj, backend="threading")(
        delayed(_predict_single_base_probability)(
            selected_df.iloc[i], saved_base_root, dataset_name, x_test
        )
        for i in range(n_rows)
    )

    base_probs = {feat: vec for feat, vec in pairs}
    return pd.DataFrame(base_probs, index=test_df.index)


def _get_deep_scaler(resolved_paths, dataset_name, schema_feature_columns):
    scaler_path = resolved_paths["deep_scaler_path"]
    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler, "loaded_saved_scaler", scaler_path

    validation_path = _require_file(
        resolved_paths["validation_probabilities_path"],
        "Validation probabilities file (for scaler reconstruction)",
    )
    val_df = pd.read_csv(validation_path)
    if "y_true" not in val_df.columns:
        raise ValueError("Validation probabilities file must contain y_true column.")

    missing = [c for c in schema_feature_columns if c not in val_df.columns]
    if missing:
        raise ValueError(
            "Cannot reconstruct scaler; validation probabilities missing %d schema features."
            % len(missing)
        )

    x_org = val_df[schema_feature_columns]
    y_org = val_df["y_true"]

    x_train, _, y_train, _ = train_test_split(
        x_org,
        y_org,
        test_size=0.2,
        stratify=y_org,
        random_state=2,
    )
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler, "reconstructed_from_validation_probabilities", validation_path


def _dataset_inference_cache(args_snapshot, parallel_jobs):
    """Load descriptors, stacked base probabilities once per dataset (batch speedup)."""
    resolved = _resolve_defaults(args_snapshot)
    features_path = _require_file(resolved["features_path"], "Features CSV")
    test_data_path = _require_file(resolved["test_data_path"], "Test data CSV")
    selected_base_csv = _require_file(
        resolved["selected_base_csv"], "Selected base artifact CSV"
    )
    schema_path = _require_file(resolved["schema_path"], "Schema JSON")

    features_df = pd.read_csv(features_path)
    if "feature" not in features_df.columns:
        raise ValueError("Features file must include a 'feature' column.")
    descriptor_columns = features_df["feature"].astype(str).tolist()

    test_df = pd.read_csv(test_data_path)
    schema = _load_json(schema_path)
    schema_feature_columns = schema.get("feature_columns", [])
    if not schema_feature_columns:
        raise ValueError("Schema JSON has no feature_columns.")

    selected_len = len(pd.read_csv(selected_base_csv))
    base_prob_df = _load_base_probabilities(
        test_df=test_df,
        feature_columns=descriptor_columns,
        selected_base_csv=selected_base_csv,
        saved_base_root=args_snapshot.saved_base_root,
        dataset_name=args_snapshot.dataset_name,
        parallel_jobs=parallel_jobs,
    )

    missing_schema_features = [
        c for c in schema_feature_columns if c not in base_prob_df.columns
    ]
    if missing_schema_features:
        raise ValueError(
            "Base probabilities missing %d schema columns. First few: %s"
            % (len(missing_schema_features), ", ".join(missing_schema_features[:10]))
        )

    return {
        "dataset_name": args_snapshot.dataset_name,
        "descriptor_columns": descriptor_columns,
        "test_df": test_df,
        "schema_feature_columns": schema_feature_columns,
        "base_prob_df": base_prob_df,
        "n_base": selected_len,
    }


def run_inference(args, dataset_cache=None):
    resolved = _resolve_defaults(args)
    deep_model_path = _require_file(resolved["deep_model_path"], "DeepAmes .h5 model")

    eff_b = getattr(args, "effective_base_jobs", None)
    if eff_b is None:
        eff_b = _effective_base_jobs(None)

    if dataset_cache is None:
        features_path = _require_file(resolved["features_path"], "Features CSV")
        test_data_path = _require_file(resolved["test_data_path"], "Test data CSV")
        selected_base_csv = _require_file(
            resolved["selected_base_csv"], "Selected base artifact CSV"
        )
        schema_path = _require_file(resolved["schema_path"], "Schema JSON")

        features_df = pd.read_csv(features_path)
        if "feature" not in features_df.columns:
            raise ValueError("Features file must include a 'feature' column.")
        descriptor_columns = features_df["feature"].astype(str).tolist()

        test_df = pd.read_csv(test_data_path)
        schema = _load_json(schema_path)
        schema_feature_columns = schema.get("feature_columns", [])
        if not schema_feature_columns:
            raise ValueError("Schema JSON has no feature_columns.")

        print("NO TRAINING PERFORMED - inference only.")
        print("Dataset: %s" % args.dataset_name)
        print("Weight: %s" % args.weight)
        print(
            "Loading %d selected base artifacts (base_jobs=%s)..."
            % (len(pd.read_csv(selected_base_csv)), eff_b)
        )

        base_prob_df = _load_base_probabilities(
            test_df=test_df,
            feature_columns=descriptor_columns,
            selected_base_csv=selected_base_csv,
            saved_base_root=args.saved_base_root,
            dataset_name=args.dataset_name,
            parallel_jobs=eff_b,
        )
    else:
        if dataset_cache["dataset_name"] != args.dataset_name:
            raise ValueError(
                "dataset_cache mismatch: cache is for %r but args.dataset_name is %r"
                % (dataset_cache["dataset_name"], args.dataset_name)
            )
        descriptor_columns = dataset_cache["descriptor_columns"]
        test_df = dataset_cache["test_df"]
        schema_feature_columns = dataset_cache["schema_feature_columns"]
        base_prob_df = dataset_cache["base_prob_df"]

        print("NO TRAINING PERFORMED - inference only.")
        print("Dataset: %s" % args.dataset_name)
        print("Weight: %s" % args.weight)
        print(
            "Reusing stacked base probabilities (%d artifacts); base_jobs skipped for cached pass."
            % dataset_cache["n_base"]
        )

    missing_schema_features = [
        c for c in schema_feature_columns if c not in base_prob_df.columns
    ]
    if missing_schema_features:
        raise ValueError(
            "Base probabilities missing %d schema columns. First few: %s"
            % (len(missing_schema_features), ", ".join(missing_schema_features[:10]))
        )

    x_meta = base_prob_df[schema_feature_columns]
    scaler, scaler_source, scaler_ref = _get_deep_scaler(
        resolved_paths=resolved,
        dataset_name=args.dataset_name,
        schema_feature_columns=schema_feature_columns,
    )
    x_meta_scaled = scaler.transform(x_meta)

    model = load_model(deep_model_path)
    pred_prob = model.predict(x_meta_scaled).reshape(-1)
    pred_class = (pred_prob > float(args.threshold)).astype(int)

    output = pd.DataFrame(index=test_df.index)
    output["id"] = test_df.index
    if "gmtamesQSAR_ID" in test_df.columns:
        output["gmtamesQSAR_ID"] = test_df["gmtamesQSAR_ID"]
    if "SMILES" in test_df.columns:
        output["SMILES"] = test_df["SMILES"]
    if "label" in test_df.columns:
        output["y_true"] = test_df["label"]

    prob_col = "prob_weight%s" % args.weight
    class_col = "class_weight%s" % args.weight
    output[prob_col] = pred_prob
    output[class_col] = pred_class

    output_path = resolved["output_csv"]
    output_dir = os.path.dirname(output_path)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)
    output.to_csv(output_path, index=False)

    print("Deep model: %s" % deep_model_path)
    print("Deep scaler: %s (%s)" % (scaler_source, scaler_ref))
    print("Rows predicted: %d" % len(output))
    print("Output CSV: %s" % output_path)


def _discover_datasets(saved_deep_root):
    if not os.path.isdir(saved_deep_root):
        raise FileNotFoundError(
            "Saved deep-model root directory not found: %s" % saved_deep_root
        )

    datasets = []
    for entry in sorted(os.listdir(saved_deep_root)):
        if not entry.startswith("results_"):
            continue
        full_path = os.path.join(saved_deep_root, entry)
        if not os.path.isdir(full_path):
            continue

        has_weight_file = False
        for filename in os.listdir(full_path):
            if filename.startswith("weight_") and filename.endswith(".h5"):
                has_weight_file = True
                break

        if has_weight_file:
            datasets.append(entry.replace("results_", "", 1))

    return datasets


def _parse_dataset_list(datasets_arg):
    if datasets_arg is None:
        return []
    return [x.strip() for x in datasets_arg.split(",") if x.strip() != ""]


def _run_batch_inference(args):
    if args.weights_start > args.weights_end:
        raise ValueError("--weights-start must be <= --weights-end.")
    if args.weights_start < 6 or args.weights_end > 18:
        raise ValueError("Batch weights must stay in [6, 18].")

    datasets = _parse_dataset_list(args.datasets)
    if len(datasets) == 0:
        if args.dataset_name is not None:
            datasets = [args.dataset_name]
        else:
            datasets = _discover_datasets(args.saved_deep_root)

    if len(datasets) == 0:
        raise ValueError(
            "No datasets discovered. Use --datasets or provide valid Saved_DeepAmes_Model_Weights."
        )

    weights = list(range(args.weights_start, args.weights_end + 1))
    failures = []
    success_count = 0
    total_runs = len(datasets) * len(weights)

    # Make sure writable output directories exist before batch execution.
    if args.batch_output_dir is not None:
        os.makedirs(args.batch_output_dir, exist_ok=True)
    else:
        os.makedirs(os.path.join(args.project_root, "Inference_Results"), exist_ok=True)

    print("Batch inference mode enabled.")
    print("Datasets: %s" % ", ".join(datasets))
    print("Weights: %s" % ", ".join([str(w) for w in weights]))
    print("Total planned runs: %d" % total_runs)
    eff_b = getattr(args, "effective_base_jobs", _effective_base_jobs(None))
    print(
        "Stacked-base parallelism (--base-jobs resolved to %s CPU threads)."
        % eff_b
    )

    for dataset_name in datasets:
        probe = argparse.Namespace(**vars(args))
        probe.dataset_name = dataset_name
        probe.weight = weights[0]

        print("-" * 80)
        print(
            "Preparing dataset=%s stacked base probs once for %d weight(s) ..."
            % (dataset_name, len(weights))
        )
        try:
            ds_cache = _dataset_inference_cache(probe, eff_b)
        except Exception as exc:
            msg = str(exc)
            print("ERROR: dataset=%s (cache/build) -> %s" % (dataset_name, msg))
            for weight in weights:
                failures.append((dataset_name, weight, msg))
            continue

        for weight in weights:
            run_args = argparse.Namespace(**vars(args))
            run_args.dataset_name = dataset_name
            run_args.weight = weight

            if args.batch_output_dir is not None:
                run_args.output_csv = os.path.join(
                    args.batch_output_dir,
                    "%s_weight%s_inference.csv" % (dataset_name, weight),
                )
            else:
                run_args.output_csv = None

            print("-" * 80)
            print("Batch run: dataset=%s weight=%s" % (dataset_name, weight))
            try:
                run_inference(run_args, dataset_cache=ds_cache)
                success_count += 1
            except Exception as exc:
                failures.append((dataset_name, weight, str(exc)))
                print(
                    "ERROR: dataset=%s weight=%s -> %s"
                    % (dataset_name, weight, str(exc))
                )

    print("=" * 80)
    print("Batch inference finished.")
    print("Succeeded: %d" % success_count)
    print("Failed: %d" % len(failures))
    if len(failures) > 0:
        print("Failed runs:")
        for dataset_name, weight, message in failures:
            print("  - %s / weight %s: %s" % (dataset_name, weight, message))

    return len(failures)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run inference on test data with saved DeepAmes/base artifacts."
    )
    parser.add_argument(
        "--dataset-name",
        required=False,
        default=None,
        help="Dataset stem, e.g. TA104_with_S9 (required for single-run mode)",
    )
    parser.add_argument("--weight", type=int, default=18, help="DeepAmes weight to load (6..18)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Classification threshold")
    parser.add_argument(
        "--base-jobs",
        type=int,
        default=None,
        help=(
            "Thread pool size for stacked-base predict_proba (joblib.threading backend). "
            "Default uses up to %d logical CPUs (capped by os.cpu_count). Use a lower integer "
            "to leave headroom, or set 1 to force serial scoring. Nested BLAS can oversubscribe: try "
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 if total CPU utilization explodes unexpectedly."
            % DEFAULT_BASE_PARALLEL_JOBS
        ),
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run batch inference across datasets and weight range",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset list for batch mode",
    )
    parser.add_argument(
        "--weights-start",
        type=int,
        default=6,
        help="Batch start weight (inclusive, default: 6)",
    )
    parser.add_argument(
        "--weights-end",
        type=int,
        default=18,
        help="Batch end weight (inclusive, default: 18)",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--project-root", default=script_dir, help="Project root path")
    parser.add_argument(
        "--saved-base-root",
        default=os.path.join(script_dir, "Saved_Base_Model_Artifacts"),
        help="Root directory containing saved base artifacts",
    )
    parser.add_argument(
        "--saved-deep-root",
        default=os.path.join(script_dir, "Saved_DeepAmes_Model_Weights"),
        help="Root directory containing saved DeepAmes .h5 models",
    )
    parser.add_argument(
        "--base-artifact-index-root",
        default=None,
        help="Root containing results_<dataset>/artifacts/selected_base_artifacts.csv "
        "(default: <project-root>/Base_Artifact_Index_CSVs)",
    )
    parser.add_argument(
        "--inference-metadata-root",
        default=None,
        help="Root containing validation schemas, validation probabilities, and optional "
        "deepames/weight_*_scaler.joblib (default: <project-root>/Inference_Metadata_Artifacts)",
    )

    parser.add_argument("--features-path", default=None, help="Override features CSV path")
    parser.add_argument("--test-data-path", default=None, help="Override test data CSV path")
    parser.add_argument("--selected-base-csv", default=None, help="Override selected_base_artifacts.csv path")
    parser.add_argument("--schema-path", default=None, help="Override validation schema JSON path")
    parser.add_argument(
        "--validation-probabilities-path",
        default=None,
        help="Override validation probabilities CSV path used for scaler reconstruction fallback",
    )
    parser.add_argument("--deep-model-path", default=None, help="Override DeepAmes .h5 path")
    parser.add_argument("--deep-scaler-path", default=None, help="Optional DeepAmes scaler .joblib path")
    parser.add_argument("--output-csv", default=None, help="Output predictions CSV path")
    parser.add_argument(
        "--batch-output-dir",
        default=None,
        help="Optional output directory override for batch mode",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    args.effective_base_jobs = _effective_base_jobs(args.base_jobs)

    if args.threshold < 0.0 or args.threshold > 1.0:
        parser.error("--threshold must be in [0, 1].")

    try:
        if args.run_all:
            failures = _run_batch_inference(args)
            if failures > 0:
                sys.exit(1)
        else:
            if args.dataset_name is None:
                parser.error("--dataset-name is required unless --run-all is used.")
            if args.weight < 6 or args.weight > 18:
                parser.error("--weight must be between 6 and 18.")
            run_inference(args)
    except Exception as exc:
        print("ERROR:", str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
