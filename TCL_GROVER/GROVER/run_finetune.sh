#!/bin/bash
# =============================================================================
# run_finetune.sh - Task-Conditioned GROVER fine-tuning
#
# Trains a single GROVER model on ALL strain+S9 combinations jointly,
# with 9-dim condition vectors (8 strain one-hot + 1 S9) concatenated
# to the encoder output before the FFN heads via --features_path.
#
# Usage:
#   ./run_finetune.sh [SEED]
# =============================================================================

set -e

SEED=${1:-42}

GROVER_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${GROVER_DIR}/data_processed"
CHECKPOINT="${GROVER_DIR}/pretrained_models/grover_large.pt"
SAVE_DIR="${GROVER_DIR}/results"
PYTHON="/home/ubuntu/anaconda3/envs/ames_grover_new/bin/python"

# Verify input files
for f in "${DATA_DIR}/train_val.csv" "${DATA_DIR}/train_val_features.npz" \
         "${DATA_DIR}/test.csv" "${DATA_DIR}/test_features.npz"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        echo "Run preprocess_data.py first."
        exit 1
    fi
done

mkdir -p "${SAVE_DIR}"

echo "=========================================="
echo "Task-Conditioned GROVER Fine-Tuning"
echo "Save dir: ${SAVE_DIR}"
echo "Seed: ${SEED}"
echo "=========================================="

cd "${GROVER_DIR}"

${PYTHON} main.py finetune \
    --data_path "${DATA_DIR}/train_val.csv" \
    --features_path "${DATA_DIR}/train_val_features.npz" \
    --separate_test_path "${DATA_DIR}/test.csv" \
    --separate_test_features_path "${DATA_DIR}/test_features.npz" \
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
    --early_stop_epoch 20 \
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
echo "Completed. Results saved to: ${SAVE_DIR}"
