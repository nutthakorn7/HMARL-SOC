#!/usr/bin/env python3
"""
HMARL-SOC Training Script
Implements Algorithm 1 from the paper: Hierarchical training with temporal abstraction.
"""

import argparse
import os
import yaml
import numpy as np
import torch
from datetime import datetime

from hmarl_soc.env.soc_env import SOCEnv
from hmarl_soc.agents.strategic_coordinator import StrategicCoordinator
from hmarl_soc.agents.threat_hunter import ThreatHunter
from hmarl_soc.agents.alert_triage import AlertTriage
from hmarl_soc.agents.response_orchestrator import ResponseOrchestrator
from hmarl_soc.core.replay_buffer import PrioritizedReplayBuffer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict, seed: int = 42, num_episodes: int = None, 
          eval_interval: int = None, save_dir: str = "checkpoints"):
    """
    Algorithm 1: HMARL-SOC Training Procedure
    
    SC updates every K steps (temporal abstraction).
    Operational agents update every step.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}, Seed: {seed}")
    
    # Seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Environment
    env = SOCEnv(config.get("environment", {}), seed=seed)
    
    # Agents
    sc_cfg = config["agents"]["strategic_coordinator"]
    th_cfg = config["agents"]["threat_hunter"]
    at_cfg = config["agents"]["alert_triage"]
    ro_cfg = config["agents"]["response_orchestrator"]
    
    K = sc_cfg.get("temporal_abstraction_K", 10)
    gamma = config["training"]["gamma"]
    
    sc = StrategicCoordinator(
        obs_dim=sc_cfg["obs_dim"], action_dim=8,
        hidden_dim=sc_cfg["hidden_dim"], lr=sc_cfg["lr"],
        gamma=gamma, clip_eps=sc_cfg["clip_eps"],
        entropy_coeff=sc_cfg["entropy_coeff"], K=K, device=device,
    )
    th = ThreatHunter(
        obs_dim=th_cfg["obs_dim"], action_dim=th_cfg["action_dim"],
        hidden_dim=th_cfg["hidden_dim"], lr=th_cfg["lr"],
        gamma=gamma, alpha=th_cfg["alpha"], tau=th_cfg["tau"], device=device,
    )
    at = AlertTriage(
        obs_dim=at_cfg["obs_dim"], num_actions=at_cfg["num_actions"],
        hidden_dim=at_cfg["hidden_dim"], lr=at_cfg["lr"],
        gamma=gamma, eps_decay=at_cfg["eps_decay"],
        target_update=at_cfg["target_update"], device=device,
    )
    ro = ResponseOrchestrator(
        obs_dim=ro_cfg["obs_dim"], action_dim=ro_cfg["action_dim"],
        hidden_dim=ro_cfg["hidden_dim"], lr_actor=ro_cfg["lr_actor"],
        lr_critic=ro_cfg["lr_critic"], gamma=gamma, tau=ro_cfg["tau"],
        device=device,
    )
    
    # Shared replay buffer
    buffer = PrioritizedReplayBuffer(
        capacity=config["training"]["replay_buffer_size"]
    )
    batch_size = config["training"]["batch_size"]
    num_episodes = num_episodes or config["training"]["num_episodes"]
    eval_interval = eval_interval or config["training"]["eval_interval"]
    
    # Logging
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_seed{seed}.csv")
    with open(log_file, "w") as f:
        f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")
    
    best_reward = -float("inf")
    episode_rewards = []
    
    print(f"Starting training: {num_episodes} episodes, K={K}")
    print("=" * 60)
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        cumulated_reward_for_sc = 0.0
        directive = None
        
        for t in range(env.max_steps):
            # --- Tier 1: SC decides every K steps ---
            if t % K == 0:
                sc_action, _ = sc.select_action(obs["sc"])
                directive = sc.get_directive(sc_action)
                
                # Store SC reward from previous K steps
                if t > 0:
                    sc.store_reward(cumulated_reward_for_sc, False)
                    cumulated_reward_for_sc = 0.0
            
            # --- Tier 2: Operational agents act every step ---
            th_action = th.select_action(obs["th"])
            at_action = at.select_action(obs["at"])
            ro_action = ro.select_action(obs["ro"])
            
            # Execute joint action
            actions = {
                "sc": directive,
                "th": th_action,
                "at": np.array([at_action]),
                "ro": ro_action,
            }
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            episode_reward += reward
            cumulated_reward_for_sc += reward
            
            # Store transition in shared buffer
            transition = {
                "obs_sc": obs["sc"], "obs_th": obs["th"],
                "obs_at": obs["at"], "obs_ro": obs["ro"],
                "act_sc": np.array([sc_action]) if directive else np.zeros(1),
                "act_th": th_action, "act_at": np.array([at_action]),
                "act_ro": ro_action,
                "reward": reward, "done": float(done),
                "next_obs_sc": next_obs["sc"], "next_obs_th": next_obs["th"],
                "next_obs_at": next_obs["at"], "next_obs_ro": next_obs["ro"],
            }
            
            # Priority: detection confidence × response urgency
            priority = abs(reward) + 0.1
            buffer.push(transition, priority)
            
            # --- Update operational agents ---
            if len(buffer) >= batch_size:
                batch, indices, weights = buffer.sample(batch_size, device)
                
                # Update TH (SAC)
                th_info = th.update(batch, weights)
                
                # Update AT (DQN)
                at_info = at.update(batch, weights)
                
                # Update RO (MADDPG) — simplified: use individual obs/actions
                all_obs = torch.cat([batch["obs_sc"], batch["obs_th"], 
                                     batch["obs_at"], batch["obs_ro"]], dim=-1)
                all_actions = torch.cat([batch["act_sc"], batch["act_th"],
                                         batch["act_at"], batch["act_ro"]], dim=-1)
                all_next_obs = torch.cat([batch["next_obs_sc"], batch["next_obs_th"],
                                          batch["next_obs_at"], batch["next_obs_ro"]], dim=-1)
                ro_info = ro.update(batch, all_obs, all_actions, all_next_obs, weights)
                
                # Update priorities
                td_errors = (np.abs(th_info["td_errors"]) + 
                            np.abs(at_info["td_errors"]) +
                            np.abs(ro_info["td_errors"])) / 3
                buffer.update_priorities(indices, td_errors)
            
            # --- Update SC every K steps ---
            if t % K == 0 and t > 0:
                sc.update(epochs=4)
            
            obs = next_obs
            if done:
                # Final SC update
                sc.store_reward(cumulated_reward_for_sc, True)
                sc.update(epochs=4)
                break
        
        # Logging
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        
        with open(log_file, "a") as f:
            f.write(f"{episode},{episode_reward:.2f},{metrics['mttd']},"
                    f"{metrics['mttr']},{metrics['fpr']:.4f},"
                    f"{int(metrics['csr'])},{metrics['compromised']}\n")
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode:6d} | Avg Reward: {avg_reward:8.2f} | "
                  f"MTTD: {metrics['mttd']:3d} | MTTR: {metrics['mttr']:3d} | "
                  f"FPR: {metrics['fpr']:.3f} | CSR: {int(metrics['csr'])}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_checkpoint(sc, th, at, ro, episode, save_dir, "best")
        
        # Periodic save
        if episode % config["training"].get("save_interval", 10000) == 0:
            save_checkpoint(sc, th, at, ro, episode, save_dir, f"ep{episode}")
    
    # Final save
    save_checkpoint(sc, th, at, ro, num_episodes, save_dir, "final")
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Logs saved to: {log_file}")
    
    return episode_rewards


def save_checkpoint(sc, th, at, ro, episode, save_dir, tag):
    path = os.path.join(save_dir, f"checkpoint_{tag}.pt")
    torch.save({
        "episode": episode,
        "sc_state": sc.network.state_dict(),
        "th_actor": th.actor.state_dict(),
        "th_critic": th.critic.state_dict(),
        "at_qnet": at.q_net.state_dict(),
        "ro_actor": ro.actor.state_dict(),
        "ro_critic": ro.critic.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="HMARL-SOC Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config, seed=args.seed, num_episodes=args.episodes,
          eval_interval=args.eval_interval, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
