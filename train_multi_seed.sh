#!/bin/bash
# Train HMARL-SOC with 4 remaining seeds on Mac M4 (MPS)
# Seed 42 already completed - only running 123, 456, 789, 1024

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SAVE_DIR="checkpoints"
EPISODES=10000

echo "============================================"
echo "HMARL-SOC Multi-Seed Training (Mac M4 MPS)"
echo "Episodes per seed: $EPISODES"
echo "Seeds: 123, 456, 789, 1024"
echo "Start time: $(date)"
echo "============================================"

for SEED in 123 456 789 1024; do
    echo ""
    echo "========================================"
    echo "Starting seed $SEED at $(date)"
    echo "========================================"
    python3 train.py --config configs/default.yaml \
        --episodes $EPISODES \
        --seed $SEED \
        --save-dir "$SAVE_DIR"
    echo "Seed $SEED complete at $(date)"
done

echo ""
echo "============================================"
echo "All seeds complete at $(date)"
echo "============================================"
echo "CSV files:"
ls -la "$SAVE_DIR"/train_seed*.csv
