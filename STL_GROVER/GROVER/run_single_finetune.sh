#!/bin/bash
# =============================================================================
# run_single_finetune.sh - Fine-tune GROVER on a single STL Ames variant
#
# Usage:
#   ./run_single_finetune.sh <VARIANT_NAME> [SEED]
#
# Examples:
#   ./run_single_finetune.sh TA100_with_S9
#   ./run_single_finetune.sh TA104_with_S9 123
# =============================================================================

set -e

VARIANT=${1:?"Usage: $0 <VARIANT_NAME> [SEED]"}
SEED=${2:-42}

GROVER_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_TRAIN_DIR="${GROVER_DIR}/data_processed/train_val"
DATA_TEST_DIR="${GROVER_DIR}/data_processed/test"
CHECKPOINT="${GROVER_DIR}/pretrained_models/grover_large.pt"
RESULTS_DIR="${GROVER_DIR}/results"
PYTHON="/home/ubuntu/anaconda3/envs/ames_grover_new/bin/python"

# Verify input files exist
if [ ! -f "${DATA_TRAIN_DIR}/${VARIANT}.csv" ]; then
    echo "ERROR: Training data not found: ${DATA_TRAIN_DIR}/${VARIANT}.csv"
    echo "Available variants:"
    ls "${DATA_TRAIN_DIR}/" | sed 's/.csv//'
    exit 1
fi

SAVE_DIR="${RESULTS_DIR}/${VARIANT}"
mkdir -p "${SAVE_DIR}"

echo "=========================================="
echo "Fine-tuning GROVER on: ${VARIANT}"
echo "Save dir: ${SAVE_DIR}"
echo "Seed: ${SEED}"
echo "=========================================="

cd "${GROVER_DIR}"

${PYTHON} main.py finetune \
    --data_path "${DATA_TRAIN_DIR}/${VARIANT}.csv" \
    --separate_test_path "${DATA_TEST_DIR}/${VARIANT}.csv" \
    --checkpoint_path "${CHECKPOINT}" \
    --dataset_type classification \
    --save_dir "${SAVE_DIR}" \
    --metric auc \
    --epochs 100 \
    --batch_size 32 \
    --dropout 0.1 \
    --ffn_hidden_size 300 \
    --ffn_num_layers 2 \
    --init_lr 1e-5 \
    --max_lr 1e-4 \
    --final_lr 1e-6 \
    --warmup_epochs 2 \
    --weight_decay 1e-5 \
    --fine_tune_coff 0.1 \
    --bond_drop_rate 0.0 \
    --early_stop_epoch 10 \
    --embedding_output_type atom \
    --split_type random \
    --split_sizes 0.9 0.1 0.0 \
    --num_folds 1 \
    --ensemble_size 1 \
    --no_features_scaling \
    --gpu 0 \
    --show_individual_scores \
    --seed "${SEED}"

echo ""
echo "Completed: ${VARIANT}"
echo "Results saved to: ${SAVE_DIR}"
