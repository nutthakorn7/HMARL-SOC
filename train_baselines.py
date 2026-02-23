#!/usr/bin/env python3
"""
Baselines for HMARL-SOC Table II comparison.

Implements 4 baselines:
  1. Rule-SOAR  — deterministic rule-based playbook (no learning)
  2. Single-DRL — single PPO agent, concatenated obs/actions
  3. IQL        — independent DQN per agent (no coordination)
  4. MAPPO      — multi-agent PPO with shared parameters

Usage:
  python3 train_baselines.py --method rule_soar --seed 42 --episodes 10000
  python3 train_baselines.py --method single_drl --seed 42 --episodes 10000
  python3 train_baselines.py --method iql --seed 42 --episodes 10000
  python3 train_baselines.py --method mappo --seed 42 --episodes 10000
"""

import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque

from hmarl_soc.env.soc_env import SOCEnv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ==============================================================
# Shared Networks
# ==============================================================

class MLP(nn.Module):
    """Simple MLP for both actors and critics."""
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """PPO-style actor-critic for discrete actions."""
    def __init__(self, obs_dim, num_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


class SimpleBuffer:
    """Uniform replay buffer."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        result = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], np.ndarray):
                result[key] = torch.FloatTensor(np.stack(vals)).to(device)
            else:
                result[key] = torch.FloatTensor(vals).to(device)
        return result

    def __len__(self):
        return len(self.buffer)


# ==============================================================
# 1. Rule-SOAR Baseline (no learning)
# ==============================================================

def train_rule_soar(config, seed, num_episodes, save_dir):
    """Deterministic rule-based SOAR playbook."""
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed)

    log_file = os.path.join(save_dir, f"train_rule_soar_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0

        for t in range(env.max_steps):
            # Rule-based actions:
            # SC: always use default directive (action 0)
            # TH: scan proportional to alert level (normalized)
            # AT: always escalate (action 0) unless low severity
            # RO: contain if compromised hosts detected
            sc_action = np.zeros(8, dtype=np.float32)
            sc_action[0] = 1.0  # default priority

            th_action = np.random.uniform(-0.5, 0.5, size=16).astype(np.float32)

            # Rule: escalate if alert queue is long, else suppress
            at_action = 0 if len(env.alert_queue) > 5 else 1

            # Rule: respond if compromised hosts > 0
            ro_action = np.ones(12, dtype=np.float32) * 0.5 if env.network.total_compromised > 0 \
                else np.zeros(12, dtype=np.float32)

            actions = {
                "sc": sc_action,
                "th": th_action,
                "at": at_action,
                "ro": ro_action
            }
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            if terminated or truncated:
                break

        metrics = info
        f.write(f"{ep},{episode_reward:.2f},{metrics.get('mttd', 200)},"
                f"{metrics.get('mttr', 200)},{metrics.get('fpr', 0):.4f},"
                f"{int(metrics.get('csr', 0))},{metrics.get('compromised', 0)}\n")

        if ep % 100 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f}")

    f.close()
    print(f"Rule-SOAR done. Saved to {log_file}")


# ==============================================================
# 2. Single-DRL Baseline (single PPO agent)
# ==============================================================

def train_single_drl(config, seed, num_episodes, save_dir):
    """Single PPO agent with concatenated obs → discretized actions."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Concatenate all obs → single obs
    total_obs = 64 + 128 + 64 + 96  # 352
    # Discretize into compound actions
    num_actions = 8 * 4  # SC strategies × AT actions = 32
    
    network = ActorCritic(total_obs, num_actions, hidden=256).to(device)
    optimizer = optim.Adam(network.parameters(), lr=3e-4)

    log_file = os.path.join(save_dir, f"train_single_drl_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    gamma = 0.99
    clip_eps = 0.2
    
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        
        ep_obs, ep_actions, ep_logprobs, ep_values, ep_rewards, ep_dones = [], [], [], [], [], []

        for t in range(env.max_steps):
            # Concatenate all observations
            cat_obs = np.concatenate([obs["sc"], obs["th"], obs["at"], obs["ro"]])
            obs_t = torch.FloatTensor(cat_obs).unsqueeze(0).to(device)
            
            action_idx, log_prob, value = network.get_action(obs_t)
            
            # Decode compound action
            sc_idx = action_idx // 4
            at_idx = action_idx % 4
            
            sc_action = np.zeros(8, dtype=np.float32)
            sc_action[sc_idx] = 1.0
            th_action = np.random.uniform(-1, 1, size=16).astype(np.float32)
            ro_action = np.random.uniform(-1, 1, size=12).astype(np.float32)
            
            actions = {"sc": sc_action, "th": th_action, "at": at_idx, "ro": ro_action}
            next_obs, reward, terminated, truncated, info = env.step(actions)

            ep_obs.append(cat_obs)
            ep_actions.append(action_idx)
            ep_logprobs.append(log_prob)
            ep_values.append(value)
            ep_rewards.append(reward)
            ep_dones.append(terminated or truncated)

            episode_reward += reward
            obs = next_obs
            if terminated or truncated:
                break

        # PPO update
        if len(ep_obs) > 1:
            returns = []
            R = 0
            for r, d in zip(reversed(ep_rewards), reversed(ep_dones)):
                R = r + gamma * R * (1 - float(d))
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(device)
            
            old_logprobs = torch.stack(ep_logprobs).detach()
            old_values = torch.cat(ep_values).detach().squeeze()
            
            obs_batch = torch.FloatTensor(np.array(ep_obs)).to(device)
            actions_batch = torch.LongTensor(ep_actions).to(device)
            
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(4):  # PPO epochs
                logits, values = network(obs_batch)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(actions_batch)
                
                ratio = torch.exp(new_logprobs - old_logprobs.squeeze())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
                entropy = dist.entropy().mean()
                
                loss = actor_loss + critic_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = info
        f.write(f"{ep},{episode_reward:.2f},{metrics.get('mttd', 200)},"
                f"{metrics.get('mttr', 200)},{metrics.get('fpr', 0):.4f},"
                f"{int(metrics.get('csr', 0))},{metrics.get('compromised', 0)}\n")

        if ep % 100 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f}")

    f.close()
    print(f"Single-DRL done. Saved to {log_file}")


# ==============================================================
# 3. IQL Baseline (Independent Q-Learning)
# ==============================================================

def train_iql(config, seed, num_episodes, save_dir):
    """Independent DQN per agent — no shared buffer, no coordination."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Per-agent Q-networks (all discrete)
    agents = {
        "sc": {"obs_dim": 64, "n_actions": 8},
        "th": {"obs_dim": 128, "n_actions": 16},  # discretize continuous
        "at": {"obs_dim": 64, "n_actions": 4},
        "ro": {"obs_dim": 96, "n_actions": 12},  # discretize continuous
    }

    q_nets = {}
    target_nets = {}
    optimizers = {}
    buffers = {}

    for name, cfg in agents.items():
        q = MLP(cfg["obs_dim"], cfg["n_actions"]).to(device)
        q_nets[name] = q
        target_nets[name] = MLP(cfg["obs_dim"], cfg["n_actions"]).to(device)
        target_nets[name].load_state_dict(q.state_dict())
        optimizers[name] = optim.Adam(q.parameters(), lr=3e-4)
        buffers[name] = SimpleBuffer(100000)

    log_file = os.path.join(save_dir, f"train_iql_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    eps_start, eps_end, eps_decay = 1.0, 0.05, 50000
    gamma = 0.99
    batch_size = 256
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0

        for t in range(env.max_steps):
            total_steps += 1
            eps = eps_end + (eps_start - eps_end) * np.exp(-total_steps / eps_decay)

            # Epsilon-greedy action selection per agent
            agent_actions = {}
            discrete_actions = {}
            for name in agents:
                if np.random.random() < eps:
                    a = np.random.randint(agents[name]["n_actions"])
                else:
                    with torch.no_grad():
                        q_vals = q_nets[name](torch.FloatTensor(obs[name]).unsqueeze(0).to(device))
                        a = q_vals.argmax(1).item()
                discrete_actions[name] = a

            # Convert discrete to env format
            sc_action = np.zeros(8, dtype=np.float32)
            sc_action[discrete_actions["sc"]] = 1.0
            
            # TH: map discrete to continuous direction
            th_action = np.zeros(16, dtype=np.float32)
            th_action[discrete_actions["th"]] = 1.0
            
            # RO: map discrete to continuous  
            ro_action = np.zeros(12, dtype=np.float32)
            ro_action[discrete_actions["ro"]] = 1.0

            actions = {
                "sc": sc_action,
                "th": th_action,
                "at": discrete_actions["at"],
                "ro": ro_action
            }
            next_obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward

            # Store transition per agent
            for name in agents:
                buffers[name].push({
                    "obs": obs[name],
                    "action": discrete_actions[name],
                    "reward": reward,
                    "next_obs": next_obs[name],
                    "done": float(terminated or truncated)
                })

            obs = next_obs
            if terminated or truncated:
                break

            # Update each agent independently
            if total_steps % 4 == 0:
                for name in agents:
                    if len(buffers[name]) < batch_size:
                        continue
                    batch = buffers[name].sample(batch_size, device)
                    
                    q_vals = q_nets[name](batch["obs"]).gather(1, batch["action"].long().unsqueeze(1))
                    with torch.no_grad():
                        next_q = target_nets[name](batch["next_obs"]).max(1)[0]
                        target = batch["reward"] + gamma * next_q * (1 - batch["done"])
                    
                    loss = nn.functional.mse_loss(q_vals.squeeze(), target)
                    optimizers[name].zero_grad()
                    loss.backward()
                    optimizers[name].step()

            # Target update
            if total_steps % 1000 == 0:
                for name in agents:
                    target_nets[name].load_state_dict(q_nets[name].state_dict())

        metrics = info
        f.write(f"{ep},{episode_reward:.2f},{metrics.get('mttd', 200)},"
                f"{metrics.get('mttr', 200)},{metrics.get('fpr', 0):.4f},"
                f"{int(metrics.get('csr', 0))},{metrics.get('compromised', 0)}\n")

        if ep % 100 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f}")

    f.close()
    print(f"IQL done. Saved to {log_file}")


# ==============================================================
# 4. MAPPO Baseline (Multi-Agent PPO, shared params)
# ==============================================================

def train_mappo(config, seed, num_episodes, save_dir):
    """Multi-Agent PPO with parameter sharing."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Shared network with max obs dim → all agents use same network
    max_obs = 128  # pad shorter obs to this
    num_actions = 16  # max actions across agents
    
    shared_net = ActorCritic(max_obs, num_actions, hidden=256).to(device)
    optimizer = optim.Adam(shared_net.parameters(), lr=3e-4)

    agent_names = ["sc", "th", "at", "ro"]
    agent_n_actions = {"sc": 8, "th": 16, "at": 4, "ro": 12}

    log_file = os.path.join(save_dir, f"train_mappo_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    gamma = 0.99
    clip_eps = 0.2

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0

        # Storage per agent
        all_data = {name: {"obs": [], "actions": [], "logprobs": [], "values": [], "rewards": [], "dones": []}
                    for name in agent_names}

        for t in range(env.max_steps):
            agent_actions_int = {}
            for name in agent_names:
                # Pad obs to max_obs
                o = obs[name]
                if len(o) < max_obs:
                    o = np.concatenate([o, np.zeros(max_obs - len(o))])
                obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
                
                action_idx, log_prob, value = shared_net.get_action(obs_t)
                # Mask to valid actions
                action_idx = action_idx % agent_n_actions[name]
                agent_actions_int[name] = action_idx
                
                all_data[name]["obs"].append(o)
                all_data[name]["actions"].append(action_idx)
                all_data[name]["logprobs"].append(log_prob)
                all_data[name]["values"].append(value)

            # Convert to env actions
            sc_action = np.zeros(8, dtype=np.float32)
            sc_action[agent_actions_int["sc"]] = 1.0
            th_action = np.zeros(16, dtype=np.float32)
            th_action[agent_actions_int["th"]] = 1.0
            ro_action = np.zeros(12, dtype=np.float32)
            ro_action[agent_actions_int["ro"]] = 1.0

            actions = {"sc": sc_action, "th": th_action, "at": agent_actions_int["at"], "ro": ro_action}
            next_obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward

            for name in agent_names:
                all_data[name]["rewards"].append(reward)
                all_data[name]["dones"].append(float(terminated or truncated))

            obs = next_obs
            if terminated or truncated:
                break

        # PPO update with all agents' data (parameter sharing)
        all_obs_batch, all_actions_batch, all_logprobs_batch = [], [], []
        all_returns_batch, all_values_batch = [], []

        for name in agent_names:
            d = all_data[name]
            if len(d["obs"]) == 0:
                continue
            
            # Compute returns
            returns = []
            R = 0
            for r, done in zip(reversed(d["rewards"]), reversed(d["dones"])):
                R = r + gamma * R * (1 - done)
                returns.insert(0, R)
            
            all_obs_batch.extend(d["obs"])
            all_actions_batch.extend(d["actions"])
            all_logprobs_batch.extend(d["logprobs"])
            all_returns_batch.extend(returns)
            all_values_batch.extend(d["values"])

        if len(all_obs_batch) > 1:
            obs_t = torch.FloatTensor(np.array(all_obs_batch)).to(device)
            acts_t = torch.LongTensor(all_actions_batch).to(device)
            old_lp = torch.stack(all_logprobs_batch).detach().squeeze()
            rets_t = torch.FloatTensor(all_returns_batch).to(device)
            old_vals = torch.cat(all_values_batch).detach().squeeze()

            advs = rets_t - old_vals
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            for _ in range(4):
                logits, values = shared_net(obs_t)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(acts_t % 16)  # mask

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advs

                loss = -torch.min(surr1, surr2).mean() + 0.5 * (rets_t - values.squeeze()).pow(2).mean() - 0.01 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = info
        f.write(f"{ep},{episode_reward:.2f},{metrics.get('mttd', 200)},"
                f"{metrics.get('mttr', 200)},{metrics.get('fpr', 0):.4f},"
                f"{int(metrics.get('csr', 0))},{metrics.get('compromised', 0)}\n")

        if ep % 100 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f}")

    f.close()
    print(f"MAPPO done. Saved to {log_file}")


# ==============================================================
# Main
# ==============================================================

METHODS = {
    "rule_soar": train_rule_soar,
    "single_drl": train_single_drl,
    "iql": train_iql,
    "mappo": train_mappo,
}

def main():
    parser = argparse.ArgumentParser(description="HMARL-SOC Baseline Training")
    parser.add_argument("--method", required=True, choices=METHODS.keys())
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    config = load_config(args.config)
    
    print(f"Training baseline: {args.method}, seed: {args.seed}, episodes: {args.episodes}")
    start = time.time()
    METHODS[args.method](config, args.seed, args.episodes, args.save_dir)
    elapsed = time.time() - start
    print(f"Completed in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
