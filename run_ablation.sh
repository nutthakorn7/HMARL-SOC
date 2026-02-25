#!/bin/bash
# HMARL-SOC Ablation Experiments
# Run all ablation variants × 5 seeds
# Each run ~46 min on MPS, running 3 at a time

SEEDS=(42 123 456 789 1024)
EPISODES=10000
WORKERS=1  # Use 1 worker per run since we run 3 in parallel

echo "========================================"
echo "HMARL-SOC Ablation Experiments"
echo "4 variants × 5 seeds = 20 runs"
echo "========================================"

# Function to run a single ablation
run_ablation() {
    local tag=$1
    local seed=$2
    local extra_args=$3
    local logfile="log_ablation_${tag}_s${seed}.log"
    
    echo "[$(date +%H:%M)] Starting ${tag} seed=${seed}"
    python3 train_fast.py \
        --episodes $EPISODES \
        --seed $seed \
        --workers $WORKERS \
        --ablation-tag "$tag" \
        $extra_args \
        > "$logfile" 2>&1
    echo "[$(date +%H:%M)] Done ${tag} seed=${seed}"
}

# ---- Variant 1: w/o Strategic Coordinator ----
echo ""
echo "=== Variant 1: w/o SC ==="
for seed in "${SEEDS[@]}"; do
    run_ablation "wo_sc" $seed "--no-sc" &
    # Run 3 at a time
    if (( $(jobs -r | wc -l) >= 3 )); then
        wait -n
    fi
done
wait
echo "=== w/o SC complete ==="

# ---- Variant 2: w/o Shared Buffer ----
echo ""
echo "=== Variant 2: w/o Shared Buffer ==="
for seed in "${SEEDS[@]}"; do
    run_ablation "wo_buf" $seed "--no-shared-buffer" &
    if (( $(jobs -r | wc -l) >= 3 )); then
        wait -n
    fi
done
wait
echo "=== w/o Shared Buffer complete ==="

# ---- Variant 3: Flat MARL (no SC + no shared buffer) ----
echo ""
echo "=== Variant 3: Flat MARL ==="
for seed in "${SEEDS[@]}"; do
    run_ablation "flat_marl" $seed "--no-sc --no-shared-buffer" &
    if (( $(jobs -r | wc -l) >= 3 )); then
        wait -n
    fi
done
wait
echo "=== Flat MARL complete ==="

# ---- Variant 4: w/o Attention Explainer ----
# Note: AttentionExplainer is not used during training (it's post-hoc).
# Results will be near-identical to Full HMARL-SOC.
# We use the full model data for this variant in the paper.

echo ""
echo "========================================"
echo "All ablation experiments complete!"
echo "CSV files in checkpoints/train_<tag>_seed<N>.csv"
echo "========================================"
