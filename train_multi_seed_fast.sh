#!/bin/bash
# HMARL-SOC Fast Multi-Seed Training (Mac M4 optimized)
# Uses train_fast.py with vectorized environments
# Runs 2 seeds in parallel (5 cores/job on 10-core M4)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SAVE_DIR="checkpoints"
EPISODES=10000
WORKERS=4  # 4 env workers per job

echo "============================================"
echo "HMARL-SOC Fast Multi-Seed Training (M4 MPS)"
echo "Episodes per seed: $EPISODES"
echo "Workers per seed: $WORKERS"
echo "Seeds: $@"
echo "Start time: $(date)"
echo "============================================"

# Default seeds if none specified
SEEDS="${@:-42 123 456 789 1024}"

# Run 2 seeds in parallel (background jobs)
PIDS=()
for SEED in $SEEDS; do
    echo ""
    echo "Starting seed $SEED at $(date)"
    python3 train_fast.py --config configs/default.yaml \
        --episodes $EPISODES \
        --seed $SEED \
        --save-dir "$SAVE_DIR" \
        --workers $WORKERS \
        > "training_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
    
    # After launching 2 jobs, wait for one to finish before launching next
    if [ ${#PIDS[@]} -ge 2 ]; then
        wait -n "${PIDS[@]}" 2>/dev/null || wait "${PIDS[0]}"
        # Remove completed PID
        NEW_PIDS=()
        for PID in "${PIDS[@]}"; do
            if kill -0 "$PID" 2>/dev/null; then
                NEW_PIDS+=("$PID")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
    fi
done

# Wait for remaining jobs
for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo ""
echo "============================================"
echo "All seeds complete at $(date)"
echo "============================================"
echo "CSV files:"
ls -la "$SAVE_DIR"/train_seed*.csv
