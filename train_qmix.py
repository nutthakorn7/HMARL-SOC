#!/usr/bin/env python3
"""
QMIX Baseline for HMARL-SOC comparison.
Implements QMIX (Rashid et al., 2018) — a value decomposition method
where a mixing network combines per-agent Q-values monotonically.

This is a FLAT multi-agent baseline (no hierarchy) for Table II comparison.
"""

import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import deque


class QNetwork(nn.Module):
    """Per-agent Q-network."""
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        return self.net(x)


class MixingNetwork(nn.Module):
    """QMIX mixing network — ensures monotonicity via abs() weights."""
    def __init__(self, n_agents, state_dim, embed_dim=64):
        super().__init__()
        self.n_agents = n_agents
        
        # Hypernetworks for mixing weights
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))
    
    def forward(self, agent_qs, state):
        """agent_qs: (batch, n_agents), state: (batch, state_dim) -> (batch, 1)"""
        batch_size = agent_qs.size(0)
        
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, -1)
        b1 = self.hyper_b1(state).unsqueeze(1)
        
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, -1, 1)
        b2 = self.hyper_b2(state).unsqueeze(1)
        
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze(-1).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        result = {}
        for key in batch[0]:
            arr = np.array([t[key] for t in batch])
            result[key] = torch.FloatTensor(arr).to(device)
        return result
    
    def __len__(self):
        return len(self.buffer)


def train_qmix(config, seed=42, num_episodes=10000, save_dir="checkpoints"):
    """Train QMIX baseline on the SOC environment."""
    device = torch.device("cpu")  # CPU avoids MPS NaN issues
    print(f"Device: {device}, Seed: {seed}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from hmarl_soc.env.soc_env import SOCEnv
    env = SOCEnv(config.get("environment", {}), seed=seed)
    
    # QMIX setup: 3 agents (TH, AT, RO) — no hierarchy
    n_agents = 3
    obs_dims = [128, 64, 96]   # TH, AT, RO observation dims
    n_actions = [16, 4, 12]    # TH(continuous→discretized), AT(discrete), RO(continuous→discretized)
    state_dim = sum(obs_dims)  # global state = concat all obs
    hidden_dim = 256
    
    # Per-agent Q-networks
    q_nets = [QNetwork(obs_dims[i], n_actions[i], hidden_dim).to(device) for i in range(n_agents)]
    target_q_nets = [copy.deepcopy(q) for q in q_nets]
    
    # Mixing network
    mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer = copy.deepcopy(mixer)
    
    # Single optimizer for all
    params = list(mixer.parameters())
    for q in q_nets:
        params += list(q.parameters())
    optimizer = optim.Adam(params, lr=3e-4)
    
    buffer = ReplayBuffer(50000)  # smaller buffer → fits M4 L2 cache
    batch_size = 512  # larger batch → better CPU throughput on M4
    gamma = 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.05, 50000
    target_update = 1000
    
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_qmix_seed{seed}.csv")
    log_handle = open(log_file, "w", buffering=8192)  # buffered writes
    log_handle.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")
    
    best_reward = -float("inf")
    episode_rewards = []
    step_count = 0
    
    print(f"Starting QMIX training: {num_episodes} episodes")
    print("=" * 60)
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        
        for t in range(env.max_steps):
            step_count += 1
            eps = eps_end + (eps_start - eps_end) * np.exp(-step_count / eps_decay)
            
            # Each agent selects action with epsilon-greedy
            actions_idx = []
            for i, q_net in enumerate(q_nets):
                obs_key = ["th", "at", "ro"][i]
                if np.random.random() < eps:
                    a = np.random.randint(n_actions[i])
                else:
                    obs_t = torch.FloatTensor(obs[obs_key]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        a = q_net(obs_t).argmax(dim=1).item()
                actions_idx.append(a)
            
            # Convert discrete indices to environment actions
            # TH: 16 bins → target specific segments with varying intensity
            th_action = np.zeros(16, dtype=np.float32)
            th_bin = actions_idx[0]
            target_seg = th_bin % 5          # which segment to focus (0-4)
            intensity = 0.3 + 0.7 * (th_bin // 5) / 3.0  # intensity 0.3-1.0
            th_action[target_seg] = intensity     # scan target segment
            th_action[5 + target_seg] = 0.5       # scope for target segment
            
            at_action = np.array([actions_idx[1]])
            
            # RO: 12 bins → target specific segments for isolate/remediate
            ro_action = np.zeros(12, dtype=np.float32)
            ro_bin = actions_idx[2]
            target_seg_ro = ro_bin % 5
            ro_intensity = 0.3 + 0.7 * (ro_bin // 5) / 2.0  # intensity 0.3-1.0
            ro_action[target_seg_ro] = ro_intensity           # isolate target
            ro_action[5 + target_seg_ro] = ro_intensity       # remediate target
            
            # No coordinator — use default directive
            env_actions = {
                "sc": np.zeros(8),  # dummy
                "th": th_action,
                "at": at_action,
                "ro": ro_action,
            }
            
            next_obs, reward, terminated, truncated, info = env.step(env_actions)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition
            transition = {
                "obs_th": obs["th"], "obs_at": obs["at"], "obs_ro": obs["ro"],
                "actions": np.array(actions_idx, dtype=np.float32),
                "reward": np.array([reward]),
                "next_obs_th": next_obs["th"], "next_obs_at": next_obs["at"], "next_obs_ro": next_obs["ro"],
                "done": np.array([float(done)]),
            }
            buffer.push(transition)
            
            # Update every 4 steps (not every step) for speed
            if len(buffer) >= batch_size and step_count % 4 == 0:
                batch = buffer.sample(batch_size, device)
                
                # Get per-agent Q-values
                agent_qs = []
                target_agent_qs = []
                for i in range(n_agents):
                    obs_key = f"obs_{['th','at','ro'][i]}"
                    next_obs_key = f"next_obs_{['th','at','ro'][i]}"
                    
                    qs = q_nets[i](batch[obs_key])
                    act_i = batch["actions"][:, i].long()
                    agent_qs.append(qs.gather(1, act_i.unsqueeze(1)).squeeze(1))
                    
                    with torch.no_grad():
                        target_qs = target_q_nets[i](batch[next_obs_key])
                        target_agent_qs.append(target_qs.max(dim=1)[0])
                
                agent_qs_stack = torch.stack(agent_qs, dim=1)
                target_qs_stack = torch.stack(target_agent_qs, dim=1)
                
                # Global state
                state = torch.cat([batch["obs_th"], batch["obs_at"], batch["obs_ro"]], dim=1)
                next_state = torch.cat([batch["next_obs_th"], batch["next_obs_at"], batch["next_obs_ro"]], dim=1)
                
                q_total = mixer(agent_qs_stack, state)
                
                with torch.no_grad():
                    target_q_total = target_mixer(target_qs_stack, next_state)
                    target = batch["reward"].squeeze(1) + gamma * (1 - batch["done"].squeeze(1)) * target_q_total
                
                loss = nn.MSELoss()(q_total, target)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()
            
            # Target update
            if step_count % target_update == 0:
                for i in range(n_agents):
                    target_q_nets[i].load_state_dict(q_nets[i].state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
            
            obs = next_obs
            if done:
                break
        
        # Log
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        
        log_handle.write(f"{episode},{episode_reward:.2f},{metrics['mttd']},"
                         f"{metrics['mttr']},{metrics['fpr']:.4f},"
                         f"{int(metrics['csr'])},{metrics['compromised']}\n")
        
        if episode % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            print(f"Episode {episode:6d} | Avg Reward: {avg_reward:8.2f} | "
                  f"MTTD: {metrics['mttd']:3d} | MTTR: {metrics['mttr']:3d} | "
                  f"FPR: {metrics['fpr']:.3f} | CSR: {int(metrics['csr'])}")
            log_handle.flush()
        
        if episode_reward > best_reward:
            best_reward = episode_reward
    
    log_handle.close()
    print(f"\nQMIX training complete. Best reward: {best_reward:.2f}")
    print(f"Logs saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="QMIX Baseline Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train_qmix(config, seed=args.seed, num_episodes=args.episodes, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
