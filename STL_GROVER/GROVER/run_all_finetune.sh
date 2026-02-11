#!/bin/bash
# =============================================================================
# run_all_finetune.sh - Fine-tune GROVER on all 16 STL Ames variants
#
# Usage:
#   ./run_all_finetune.sh [SEED]
# =============================================================================

set -e

SEED=${1:-42}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VARIANTS=(
    "TA100_with_S9"
    "TA100_without_S9"
    "TA102_with_S9"
    "TA102_without_S9"
    "TA104_with_S9"
    "TA104_without_S9"
    "TA1535_with_S9"
    "TA1535_without_S9"
    "TA1537_with_S9"
    "TA1537_without_S9"
    "TA1538_with_S9"
    "TA1538_without_S9"
    "TA97_with_S9"
    "TA97_without_S9"
    "TA98_with_S9"
    "TA98_without_S9"
)

TOTAL=${#VARIANTS[@]}
CURRENT=0

echo "============================================"
echo "GROVER Fine-Tuning: All ${TOTAL} STL Variants"
echo "Seed: ${SEED}"
echo "============================================"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[${CURRENT}/${TOTAL}] Starting: ${VARIANT}"
    bash "${SCRIPT_DIR}/run_single_finetune.sh" "${VARIANT}" "${SEED}"
    echo ""
done

echo "============================================"
echo "All ${TOTAL} variants completed."
echo "Results in: ${SCRIPT_DIR}/results/"
echo "============================================"
