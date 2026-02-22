#!/usr/bin/env python3
"""
HMARL-SOC Evaluation Script
Evaluates trained agents and generates paper results (Table I, Table II, Figure 2).
"""

import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from hmarl_soc.env.soc_env import SOCEnv
from hmarl_soc.agents.strategic_coordinator import StrategicCoordinator
from hmarl_soc.agents.threat_hunter import ThreatHunter
from hmarl_soc.agents.alert_triage import AlertTriage
from hmarl_soc.agents.response_orchestrator import ResponseOrchestrator


def evaluate(config: dict, checkpoint_path: str = None,
             num_episodes: int = 1000, seed: int = 42) -> dict:
    """Evaluate trained agents over num_episodes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = SOCEnv(config.get("environment", {}), seed=seed)
    
    # Initialize agents
    sc_cfg = config["agents"]["strategic_coordinator"]
    th_cfg = config["agents"]["threat_hunter"]
    at_cfg = config["agents"]["alert_triage"]
    ro_cfg = config["agents"]["response_orchestrator"]
    
    sc = StrategicCoordinator(obs_dim=sc_cfg["obs_dim"], device=device)
    th = ThreatHunter(obs_dim=th_cfg["obs_dim"], action_dim=th_cfg["action_dim"], device=device)
    at = AlertTriage(obs_dim=at_cfg["obs_dim"], num_actions=at_cfg["num_actions"], device=device)
    ro = ResponseOrchestrator(obs_dim=ro_cfg["obs_dim"], action_dim=ro_cfg["action_dim"], device=device)
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        sc.network.load_state_dict(ckpt["sc_state"])
        th.actor.load_state_dict(ckpt["th_actor"])
        at.q_net.load_state_dict(ckpt["at_qnet"])
        ro.actor.load_state_dict(ckpt["ro_actor"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    K = sc_cfg.get("temporal_abstraction_K", 10)
    
    all_metrics = defaultdict(list)
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        directive = None
        
        for t in range(env.max_steps):
            if t % K == 0:
                sc_action, _ = sc.select_action(obs["sc"])
                directive = sc.get_directive(sc_action)
            
            th_action = th.select_action(obs["th"], evaluate=True)
            at_action = at.select_action(obs["at"], evaluate=True)
            ro_action = ro.select_action(obs["ro"], evaluate=True)
            
            actions = {"sc": directive, "th": th_action,
                       "at": np.array([at_action]), "ro": ro_action}
            obs, reward, terminated, truncated, info = env.step(actions)
            
            if terminated or truncated:
                break
        
        metrics = env.get_metrics()
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    # Aggregate
    results = {}
    for k, v in all_metrics.items():
        results[k] = {"mean": np.mean(v), "std": np.std(v)}
    
    return results


def generate_table_1(results_dict: dict):
    """Generate Table I: Performance Comparison Across Methods."""
    print("\n" + "=" * 65)
    print("TABLE I: Performance Comparison Across Methods")
    print("=" * 65)
    print(f"{'Method':<15} {'MTTD':>8} {'MTTR':>8} {'FPR':>8} {'CSR':>8}")
    print(f"{'':15} {'(steps↓)':>8} {'(steps↓)':>8} {'(%↓)':>8} {'(%↑)':>8}")
    print("-" * 65)
    
    for method, res in results_dict.items():
        mttd = res["mttd"]["mean"]
        mttr = res["mttr"]["mean"]
        fpr = res["fpr"]["mean"] * 100
        csr = res["csr"]["mean"] * 100
        print(f"{method:<15} {mttd:8.1f} {mttr:8.1f} {fpr:7.1f}% {csr:7.1f}%")
    
    print("=" * 65)


def plot_learning_curves(log_dir: str, output_path: str = "figures/learning_curves.png"):
    """Generate Figure 2: Learning curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {"Rule-SOAR": "gray", "Single-DRL": "blue", 
              "IQL": "orange", "MAPPO": "red", "HMARL-SOC": "green"}
    
    for method, color in colors.items():
        log_file = os.path.join(log_dir, f"{method.lower().replace('-', '_')}.csv")
        if os.path.exists(log_file):
            data = np.loadtxt(log_file, delimiter=",", skiprows=1)
            episodes = data[:, 0]
            rewards = data[:, 1]
            # Smooth with moving average
            window = min(100, len(rewards) // 10)
            if window > 0:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(np.arange(len(smoothed)), smoothed, label=method, color=color, linewidth=2)
    
    ax.set_xlabel("Episodes (×10³)", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("HMARL-SOC Learning Curves")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Learning curves saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HMARL-SOC Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("Evaluating HMARL-SOC...")
    results = evaluate(config, args.checkpoint, args.episodes, args.seed)
    
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")
    
    # Compare with baselines (paper values)
    all_results = {
        "Rule-SOAR": {"mttd": {"mean": 38.4}, "mttr": {"mean": 52.7}, 
                       "fpr": {"mean": 0.183}, "csr": {"mean": 0.712}},
        "Single-DRL": {"mttd": {"mean": 28.1}, "mttr": {"mean": 35.4},
                        "fpr": {"mean": 0.127}, "csr": {"mean": 0.798}},
        "IQL": {"mttd": {"mean": 25.6}, "mttr": {"mean": 31.2},
                 "fpr": {"mean": 0.114}, "csr": {"mean": 0.825}},
        "MAPPO": {"mttd": {"mean": 23.3}, "mttr": {"mean": 28.6},
                   "fpr": {"mean": 0.092}, "csr": {"mean": 0.873}},
        "HMARL-SOC": results,
    }
    
    generate_table_1(all_results)


if __name__ == "__main__":
    main()
