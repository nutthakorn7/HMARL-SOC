#!/usr/bin/env python3
"""Generate training result plots for HMARL-SOC."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(csv_path, output_dir='.'):
    df = pd.read_csv(csv_path)
    
    # Smooth with rolling window
    window = 100
    df_smooth = df.rolling(window=window, min_periods=1).mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('HMARL-SOC Training Results (10K Episodes, Seed 42)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Reward
    ax = axes[0, 0]
    ax.plot(df['episode'], df['reward'], alpha=0.15, color='blue')
    ax.plot(df['episode'], df_smooth['reward'], color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Cumulative Reward')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    # 2. MTTD
    ax = axes[0, 1]
    ax.plot(df['episode'], df['mttd'], alpha=0.15, color='green')
    ax.plot(df['episode'], df_smooth['mttd'], color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('MTTD (steps)')
    ax.set_title('Mean Time to Detect (↓ better)')
    ax.grid(True, alpha=0.3)
    
    # 3. FPR
    ax = axes[1, 0]
    ax.plot(df['episode'], df['fpr'], alpha=0.15, color='orange')
    ax.plot(df['episode'], df_smooth['fpr'], color='orange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('FPR (%)')
    ax.set_title('False Positive Rate (↓ better)')
    ax.grid(True, alpha=0.3)
    
    # 4. CSR
    ax = axes[1, 1]
    ax.plot(df['episode'], df['csr'], alpha=0.15, color='red')
    ax.plot(df['episode'], df_smooth['csr'], color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('CSR')
    ax.set_title('Containment Success Rate (↑ better)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_path = "checkpoints/train_seed42.csv"
    plot_results(csv_path, output_dir=".")
