#!/bin/bash
# =============================================================================
# gmtames Reproduction Script
# Reproduces Table 2: Averaged Out-of-Domain Test Set Performance
# Across 16 S. typhimurium +/- S9 Strain Tasks
# =============================================================================

# Run from this script's directory (project root) so paths resolve correctly
cd "$(dirname "$0")"

# Set test split strategy
TESTSPLIT="scaffold"

# Set experiment name
EXPERIMENT="mtg_experiment_${TESTSPLIT}"

# Define all task groupings
# -----------------------------------------------------------------------------

# STL (Single Task Learning) - 16 individual strain models
STL_TASKS=(
    "TA100"
    "TA100_S9"
    "TA102"
    "TA102_S9"
    "TA104"
    "TA104_S9"
    "TA1535"
    "TA1535_S9"
    "TA1537"
    "TA1537_S9"
    "TA1538"
    "TA1538_S9"
    "TA97"
    "TA97_S9"
    "TA98"
    "TA98_S9"
)

# uMTL (Ungrouped Multitask Learning) - all 16 strains together
UMTL_TASK="TA100,TA100_S9,TA102,TA102_S9,TA104,TA104_S9,TA1535,TA1535_S9,TA1537,TA1537_S9,TA1538,TA1538_S9,TA97,TA97_S9,TA98,TA98_S9"

# gMTL (Grouped Multitask Learning) - mechanistic groupings
GMTL_TASKS=(
    # Substitution strains (all)
    "TA100,TA100_S9,TA102,TA102_S9,TA104,TA104_S9,TA1535,TA1535_S9"
    # Frameshift strains (all)
    "TA1537,TA1537_S9,TA1538,TA1538_S9,TA97,TA97_S9,TA98,TA98_S9"
    # Non-S9 strains
    "TA100,TA102,TA104,TA1535,TA1537,TA1538,TA97,TA98"
    # S9 strains
    "TA100_S9,TA102_S9,TA104_S9,TA1535_S9,TA1537_S9,TA1538_S9,TA97_S9,TA98_S9"
    # Non-S9 substitution strains
    "TA100,TA102,TA104,TA1535"
    # Non-S9 frameshift strains
    "TA1537,TA1538,TA97,TA98"
    # S9 substitution strains
    "TA100_S9,TA102_S9,TA104_S9,TA1535_S9"
    # S9 frameshift strains
    "TA1537_S9,TA1538_S9,TA97_S9,TA98_S9"
)

# =============================================================================
# Run Experiments
# =============================================================================

echo "========================================"
echo "gmtames Reproduction Pipeline"
echo "Test Split: ${TESTSPLIT}"
echo "========================================"

# Step 0: Generate base datasets from master data (gmtames/data/smiles_to_fp/)
# Ensure gmtamesQSAR_fingerprints.csv and gmtamesQSAR_endpoints_scaffold.csv exist there first (run smiles_to_morgan_fingerprint.py if needed)
echo ""
echo "[Step 0/5] Generating base datasets from master data (smiles_to_fp)..."
python -m gmtames data --generateBaseDatasets --testsplit ${TESTSPLIT} --output ${EXPERIMENT}

# Step 1: Train STL models (16 single-task models)
echo ""
echo "[Step 1/5] Training STL models (16 strains)..."
for task in "${STL_TASKS[@]}"; do
    echo "  Training STL model for: ${task}"
    python -m gmtames mtg --tasks ${task} --testsplit ${TESTSPLIT} --output ${EXPERIMENT} --device mps
done

# Step 2: Train uMTL model (1 model with all 16 tasks)
echo ""
echo "[Step 2/5] Training uMTL model (all 16 strains)..."
python -m gmtames mtg --tasks ${UMTL_TASK} --testsplit ${TESTSPLIT} --output ${EXPERIMENT} --device mps

# Step 3: Train gMTL models (8 mechanistic groupings)
echo ""
echo "[Step 3/5] Training gMTL models (8 groupings)..."
for task in "${GMTL_TASKS[@]}"; do
    echo "  Training gMTL model for: ${task}"
    python -m gmtames mtg --tasks ${task} --testsplit ${TESTSPLIT} --output ${EXPERIMENT} --device mps
done

# Step 4: Calculate bootstrapped results
echo ""
echo "[Step 4/5] Calculating bootstrapped test results..."
python -m gmtames results ${EXPERIMENT}

echo ""
echo "========================================"
echo "Experiment Complete!"
echo "Results saved to: output/${EXPERIMENT}/test_results/"
echo "========================================"
echo ""
echo "Key output files:"
echo "  - gmtames_curated_averaged_results.csv  (Table 2 metrics)"
echo "  - gmtames_curated_task_results.csv      (Per-strain metrics)"
echo "  - gmtames_full_results.csv              (All bootstrap statistics)"
