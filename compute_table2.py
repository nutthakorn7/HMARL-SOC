#!/usr/bin/env python3
"""Compute final Table II from multi-seed CSV data."""
import numpy as np
import os

SEEDS = [42, 123, 456, 789, 1024]
CKPT_DIR = "checkpoints"

# Method -> CSV prefix mapping
METHODS = {
    "Rule-SOAR": "train_rule_soar_seed",
    "Single-DRL": "train_single_drl_seed",
    "IQL": "train_iql_seed",
    "MAPPO": "train_mappo_seed",
    "QMIX": "train_qmix_seed",
    "HMARL-SOC": "train_seed",
}

# How many episodes from the END to average over
EVAL_WINDOW = 2000

print("=" * 80)
print("TABLE II: Performance Comparison (last 2000 episodes, 5 seeds)")
print("=" * 80)
print(f"{'Method':<15} {'Reward':>15} {'MTTD':>15} {'MTTR':>15} {'FPR (%)':>15} {'CSR (%)':>15}")
print("-" * 95)

for method, prefix in METHODS.items():
    all_reward, all_mttd, all_mttr, all_fpr, all_csr = [], [], [], [], []
    
    for seed in SEEDS:
        csv_path = os.path.join(CKPT_DIR, f"{prefix}{seed}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found!")
            continue
        
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        # Take last EVAL_WINDOW episodes
        tail = data[-EVAL_WINDOW:]
        
        all_reward.append(np.mean(tail["reward"]))
        all_mttd.append(np.mean(tail["mttd"]))
        all_mttr.append(np.mean(tail["mttr"]))
        all_fpr.append(np.mean(tail["fpr"]) * 100)  # Convert to %
        all_csr.append(np.mean(tail["csr"]) * 100)   # Convert to %
    
    r = np.array(all_reward)
    mttd = np.array(all_mttd)
    mttr = np.array(all_mttr)
    fpr = np.array(all_fpr)
    csr = np.array(all_csr)
    
    print(f"{method:<15} "
          f"{r.mean():+7.1f}±{r.std():4.1f}  "
          f"{mttd.mean():7.1f}±{mttd.std():4.1f}  "
          f"{mttr.mean():7.1f}±{mttr.std():5.1f}  "
          f"{fpr.mean():7.2f}±{fpr.std():4.2f}  "
          f"{csr.mean():7.1f}±{csr.std():5.1f}")

print("=" * 95)

# Also print per-seed detail for HMARL-SOC and QMIX (the most interesting ones)
for method in ["HMARL-SOC", "QMIX"]:
    prefix = METHODS[method]
    print(f"\n--- {method} per-seed detail (last {EVAL_WINDOW} eps) ---")
    for seed in SEEDS:
        csv_path = os.path.join(CKPT_DIR, f"{prefix}{seed}.csv")
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        tail = data[-EVAL_WINDOW:]
        print(f"  seed {seed:4d}: Reward={np.mean(tail['reward']):+7.2f}  "
              f"MTTD={np.mean(tail['mttd']):6.1f}  MTTR={np.mean(tail['mttr']):6.1f}  "
              f"FPR={np.mean(tail['fpr'])*100:5.2f}%  CSR={np.mean(tail['csr'])*100:5.1f}%")
