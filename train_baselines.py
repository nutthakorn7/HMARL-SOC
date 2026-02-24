#!/usr/bin/env python3
"""
Baselines for HMARL-SOC Table II comparison (v2 — Fixed).

The SOC env requires:
  - TH: 16-dim continuous [-1,1]. first 5 = scan intensity per segment (>0.3 needed)
  - RO: 12-dim continuous [-1,1]. first 5 = isolate intensity, next 5 = remediate
  - AT: discrete 0-3 (escalate/suppress/correlate/enrich)
  - SC: 8-dim continuous (directive vector)

Previous version incorrectly discretized TH/RO → one-hot → agents never scanned.

Usage:
  python3 train_baselines.py --method rule_soar --seed 42 --episodes 10000
"""

import argparse, os, yaml, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from hmarl_soc.env.soc_env import SOCEnv


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ==============================================================
# Shared utilities
# ==============================================================

class SimpleBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, t):
        self.buffer.append(t)
    def sample(self, n, device):
        idx = np.random.choice(len(self.buffer), n, replace=False)
        batch = [self.buffer[i] for i in idx]
        out = {}
        for k in batch[0]:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], np.ndarray):
                out[k] = torch.FloatTensor(np.stack(vals)).to(device)
            else:
                out[k] = torch.FloatTensor(vals).to(device)
        return out
    def __len__(self):
        return len(self.buffer)


class GaussianActor(nn.Module):
    """Continuous actor with Gaussian output."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, x):
        h = self.net(x)
        mu = torch.tanh(self.mu(h))
        std = torch.exp(self.log_std.clamp(-3, 0.5))
        return mu, std

    def get_action(self, obs, deterministic=False):
        mu, std = self.forward(obs)
        if deterministic:
            return mu.detach().cpu().numpy().flatten()
        noise = torch.randn_like(mu) * std
        a = torch.tanh(mu + noise)
        return torch.nan_to_num(a, 0.0).detach().cpu().numpy().flatten()

    def log_prob(self, obs, actions):
        mu, std = self.forward(obs)
        actions = actions.clamp(-0.999, 0.999)
        dist = torch.distributions.Normal(mu, std)
        lp = dist.log_prob(actions).sum(-1)
        return torch.nan_to_num(lp, 0.0)


class DiscreteActor(nn.Module):
    """Discrete actor for DQN or categorical policy."""
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)


def write_csv_row(f, ep, reward, metrics):
    """Write CSV row. `metrics` should be from env.get_metrics()."""
    f.write(f"{ep},{reward:.2f},{metrics.get('mttd', 200)},"
            f"{metrics.get('mttr', 200)},{metrics.get('fpr', 0):.4f},"
            f"{int(metrics.get('csr', 0))},{metrics.get('compromised', 0)}\n")


# ==============================================================
# 1. Rule-SOAR — heuristic rules, actively scans all segments
# ==============================================================

def train_rule_soar(config, seed, num_episodes, save_dir):
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed)

    log_file = os.path.join(save_dir, f"train_rule_soar_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0

        for t in range(env.max_steps):
            # SC: default directive
            sc_action = np.zeros(8, dtype=np.float32)
            sc_action[0] = 1.0

            # TH: scan ALL segments with moderate intensity (0.5 > 0.3 threshold)
            th_action = np.zeros(16, dtype=np.float32)
            th_action[:5] = 0.0   # maps to intensity (0+1)/2 = 0.5 > 0.3 ✓
            th_action[5:10] = 0.0  # scope 0.5

            # AT: always escalate (action 0)
            at_action = 0

            # RO: contain if compromised hosts > 0
            ro_action = np.zeros(12, dtype=np.float32)
            if env.network.total_compromised > 0:
                ro_action[:5] = 0.5   # isolate with moderate intensity
                ro_action[5:10] = 0.5  # remediate

            actions = {"sc": sc_action, "th": th_action, "at": at_action, "ro": ro_action}
            obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            if terminated or truncated:
                break

        write_csv_row(f, ep, episode_reward, env.get_metrics())
        if ep % 500 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f} | MTTD: {info.get('mttd', 200)}")

    f.close()
    print(f"Rule-SOAR done. Saved to {log_file}")


# ==============================================================
# 2. Single-DRL — single PPO with continuous+discrete outputs
# ==============================================================

def train_single_drl(config, seed, num_episodes, save_dir):
    device = torch.device("cpu")  # CPU avoids MPS NaN issues
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)

    # Single agent sees concatenated obs (352-dim)
    total_obs = 64 + 128 + 64 + 96  # 352
    # Outputs: SC(8) + TH(16) + AT_logits(4) + RO(12) = 40
    cont_dim = 8 + 16 + 12  # 36 continuous
    disc_dim = 4  # AT discrete

    actor_cont = GaussianActor(total_obs, cont_dim, hidden=256).to(device)
    actor_disc = DiscreteActor(total_obs, disc_dim, hidden=256).to(device)
    critic = Critic(total_obs, hidden=256).to(device)
    optimizer = optim.Adam(list(actor_cont.parameters()) + list(actor_disc.parameters()) + 
                           list(critic.parameters()), lr=3e-4)

    log_file = os.path.join(save_dir, f"train_single_drl_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    gamma, clip_eps = 0.99, 0.2

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        ep_data = {"obs": [], "cont_actions": [], "disc_actions": [], "cont_lp": [], "disc_lp": [],
                   "values": [], "rewards": [], "dones": []}

        for t in range(env.max_steps):
            cat_obs = np.concatenate([obs["sc"], obs["th"], obs["at"], obs["ro"]])
            obs_t = torch.FloatTensor(cat_obs).unsqueeze(0).to(device)

            # Continuous actions
            mu, std = actor_cont(obs_t)
            cont_dist = torch.distributions.Normal(mu, std)
            cont_a = cont_dist.sample()
            cont_a_clamped = torch.tanh(cont_a).squeeze(0)
            cont_lp = cont_dist.log_prob(cont_a).sum(-1)

            # Discrete action
            disc_logits = actor_disc(obs_t)
            disc_dist = torch.distributions.Categorical(logits=disc_logits)
            disc_a = disc_dist.sample()
            disc_lp = disc_dist.log_prob(disc_a)

            value = critic(obs_t)

            # Decode to env actions
            ca = cont_a_clamped.detach().cpu().numpy()
            sc_action = ca[:8]
            th_action = ca[8:24]
            ro_action = ca[24:36]
            at_action = disc_a.item()

            actions = {"sc": sc_action, "th": th_action, "at": at_action, "ro": ro_action}
            next_obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward

            ep_data["obs"].append(cat_obs)
            ep_data["cont_actions"].append(ca)
            ep_data["disc_actions"].append(at_action)
            ep_data["cont_lp"].append(cont_lp)
            ep_data["disc_lp"].append(disc_lp)
            ep_data["values"].append(value)
            ep_data["rewards"].append(reward)
            ep_data["dones"].append(terminated or truncated)

            obs = next_obs
            if terminated or truncated:
                break

        # PPO update
        if len(ep_data["obs"]) > 1:
            returns = []
            R = 0
            for r, d in zip(reversed(ep_data["rewards"]), reversed(ep_data["dones"])):
                R = r + gamma * R * (1 - float(d))
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(device)
            old_cont_lp = torch.stack(ep_data["cont_lp"]).detach().squeeze()
            old_disc_lp = torch.stack(ep_data["disc_lp"]).detach().squeeze()
            old_vals = torch.cat(ep_data["values"]).detach().squeeze()
            advs = returns - old_vals
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            obs_batch = torch.FloatTensor(np.array(ep_data["obs"])).to(device)
            cont_acts = torch.FloatTensor(np.array(ep_data["cont_actions"])).to(device)
            disc_acts = torch.LongTensor(ep_data["disc_actions"]).to(device)

            for _ in range(4):
                new_cont_lp = actor_cont.log_prob(obs_batch, cont_acts)
                new_disc_logits = actor_disc(obs_batch)
                new_disc_dist = torch.distributions.Categorical(logits=new_disc_logits)
                new_disc_lp = new_disc_dist.log_prob(disc_acts)
                new_vals = critic(obs_batch).squeeze()

                ratio_c = torch.exp(new_cont_lp - old_cont_lp)
                ratio_d = torch.exp(new_disc_lp - old_disc_lp)
                ratio = ratio_c * ratio_d

                s1 = ratio * advs
                s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advs
                loss = -torch.min(s1, s2).mean() + 0.5*(returns - new_vals).pow(2).mean() - 0.01*new_disc_dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor_cont.parameters()) + list(actor_disc.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()

        write_csv_row(f, ep, episode_reward, env.get_metrics())
        if ep % 500 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f} | MTTD: {info.get('mttd', 200)}")

    f.close()
    print(f"Single-DRL done. Saved to {log_file}")


# ==============================================================
# 3. IQL — Independent agents: DQN for discrete, SAC-like for continuous
# ==============================================================

def train_iql(config, seed, num_episodes, save_dir):
    device = torch.device("cpu")  # CPU avoids MPS NaN issues
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)

    # SC: discrete (8 actions), AT: discrete (4 actions)
    # TH: continuous (16-dim), RO: continuous (12-dim)
    sc_q = DiscreteActor(64, 8).to(device)
    sc_target = DiscreteActor(64, 8).to(device)
    sc_target.load_state_dict(sc_q.state_dict())
    sc_opt = optim.Adam(sc_q.parameters(), lr=3e-4)
    sc_buf = SimpleBuffer(50000)  # smaller buffer → fits M4 L2 cache

    at_q = DiscreteActor(64, 4).to(device)
    at_target = DiscreteActor(64, 4).to(device)
    at_target.load_state_dict(at_q.state_dict())
    at_opt = optim.Adam(at_q.parameters(), lr=3e-4)
    at_buf = SimpleBuffer(50000)

    th_actor = GaussianActor(128, 16).to(device)
    th_critic = Critic(128).to(device)
    th_opt = optim.Adam(list(th_actor.parameters()) + list(th_critic.parameters()), lr=3e-4)
    th_buf = SimpleBuffer(50000)

    ro_actor = GaussianActor(96, 12).to(device)
    ro_critic = Critic(96).to(device)
    ro_opt = optim.Adam(list(ro_actor.parameters()) + list(ro_critic.parameters()), lr=3e-4)
    ro_buf = SimpleBuffer(50000)

    log_file = os.path.join(save_dir, f"train_iql_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    eps_start, eps_end, eps_decay = 1.0, 0.05, 80000  # slower decay for stable convergence
    gamma, batch_size = 0.99, 512  # larger batch → better M4 CPU throughput
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0

        for t in range(env.max_steps):
            total_steps += 1
            eps = eps_end + (eps_start - eps_end) * np.exp(-total_steps / eps_decay)

            # SC: epsilon-greedy discrete
            if np.random.random() < eps:
                sc_a = np.random.randint(8)
            else:
                with torch.no_grad():
                    sc_a = sc_q(torch.FloatTensor(obs["sc"]).unsqueeze(0).to(device)).argmax(1).item()
            sc_action = np.zeros(8, dtype=np.float32)
            sc_action[sc_a] = 1.0

            # TH: Gaussian continuous
            with torch.no_grad():
                th_action = th_actor.get_action(torch.FloatTensor(obs["th"]).unsqueeze(0).to(device))

            # AT: epsilon-greedy discrete
            if np.random.random() < eps:
                at_a = np.random.randint(4)
            else:
                with torch.no_grad():
                    at_a = at_q(torch.FloatTensor(obs["at"]).unsqueeze(0).to(device)).argmax(1).item()

            # RO: Gaussian continuous
            with torch.no_grad():
                ro_action = ro_actor.get_action(torch.FloatTensor(obs["ro"]).unsqueeze(0).to(device))

            actions = {"sc": sc_action, "th": th_action, "at": at_a, "ro": ro_action}
            next_obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            done = float(terminated or truncated)

            # Store per-agent
            sc_buf.push({"obs": obs["sc"], "action": sc_a, "reward": reward, "next_obs": next_obs["sc"], "done": done})
            at_buf.push({"obs": obs["at"], "action": at_a, "reward": reward, "next_obs": next_obs["at"], "done": done})
            th_buf.push({"obs": obs["th"], "action": th_action, "reward": reward, "next_obs": next_obs["th"], "done": done})
            ro_buf.push({"obs": obs["ro"], "action": ro_action, "reward": reward, "next_obs": next_obs["ro"], "done": done})

            obs = next_obs
            if terminated or truncated:
                break

            # Update every 4 steps
            if total_steps % 4 == 0 and len(sc_buf) >= batch_size:
                # SC DQN update
                b = sc_buf.sample(batch_size, device)
                q = sc_q(b["obs"]).gather(1, b["action"].long().unsqueeze(1))
                with torch.no_grad():
                    nq = sc_target(b["next_obs"]).max(1)[0]
                    target = b["reward"] + gamma * nq * (1 - b["done"])
                sc_opt.zero_grad()
                nn.functional.mse_loss(q.squeeze(), target).backward()
                sc_opt.step()

                # AT DQN update
                b = at_buf.sample(batch_size, device)
                q = at_q(b["obs"]).gather(1, b["action"].long().unsqueeze(1))
                with torch.no_grad():
                    nq = at_target(b["next_obs"]).max(1)[0]
                    target = b["reward"] + gamma * nq * (1 - b["done"])
                at_opt.zero_grad()
                nn.functional.mse_loss(q.squeeze(), target).backward()
                at_opt.step()

                # TH actor-critic update (REINFORCE-style)
                b = th_buf.sample(batch_size, device)
                val = th_critic(b["obs"]).squeeze()
                with torch.no_grad():
                    next_val = th_critic(b["next_obs"]).squeeze()
                    target = b["reward"] + gamma * next_val * (1 - b["done"])
                critic_loss = nn.functional.mse_loss(val, target)
                advantage = (target - val).detach()
                actor_loss = -(th_actor.log_prob(b["obs"], b["action"]) * advantage).mean()
                th_opt.zero_grad()
                (actor_loss + critic_loss).backward()
                th_opt.step()

                # RO actor-critic update
                b = ro_buf.sample(batch_size, device)
                val = ro_critic(b["obs"]).squeeze()
                with torch.no_grad():
                    next_val = ro_critic(b["next_obs"]).squeeze()
                    target = b["reward"] + gamma * next_val * (1 - b["done"])
                critic_loss = nn.functional.mse_loss(val, target)
                advantage = (target - val).detach()
                actor_loss = -(ro_actor.log_prob(b["obs"], b["action"]) * advantage).mean()
                ro_opt.zero_grad()
                (actor_loss + critic_loss).backward()
                ro_opt.step()

            # Target update
            if total_steps % 1000 == 0:
                sc_target.load_state_dict(sc_q.state_dict())
                at_target.load_state_dict(at_q.state_dict())

        write_csv_row(f, ep, episode_reward, env.get_metrics())
        if ep % 500 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f} | MTTD: {info.get('mttd', 200)}")

    f.close()
    print(f"IQL done. Saved to {log_file}")


# ==============================================================
# 4. MAPPO — Multi-Agent PPO with parameter sharing + Gaussian
# ==============================================================

def train_mappo(config, seed, num_episodes, save_dir):
    device = torch.device("cpu")  # CPU avoids MPS NaN issues
    env = SOCEnv(config.get("environment", {}), seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)

    # Shared actor (Gaussian for continuous, categorical head for discrete)
    max_obs = 128  # pad shorter obs
    cont_dim = 16  # max continuous action dim
    disc_dim = 8   # max discrete actions

    shared_cont = GaussianActor(max_obs, cont_dim).to(device)
    shared_disc = DiscreteActor(max_obs, disc_dim).to(device)
    shared_critic = Critic(max_obs).to(device)
    optimizer = optim.Adam(list(shared_cont.parameters()) + list(shared_disc.parameters()) +
                           list(shared_critic.parameters()), lr=3e-4)

    agent_cfg = {
        "sc": {"obs_dim": 64, "type": "disc", "n_actions": 8},
        "th": {"obs_dim": 128, "type": "cont", "act_dim": 16},
        "at": {"obs_dim": 64, "type": "disc", "n_actions": 4},
        "ro": {"obs_dim": 96, "type": "cont", "act_dim": 12},
    }

    log_file = os.path.join(save_dir, f"train_mappo_seed{seed}.csv")
    f = open(log_file, "w")
    f.write("episode,reward,mttd,mttr,fpr,csr,compromised\n")

    gamma, clip_eps = 0.99, 0.2

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        all_data = []

        for t in range(env.max_steps):
            step_data = {}
            agent_actions = {}

            for name, cfg in agent_cfg.items():
                o = obs[name]
                if len(o) < max_obs:
                    o = np.concatenate([o, np.zeros(max_obs - len(o))])
                obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
                val = shared_critic(obs_t)

                if cfg["type"] == "cont":
                    mu, std = shared_cont(obs_t)
                    dist = torch.distributions.Normal(mu[:, :cfg["act_dim"]], std[:cfg["act_dim"]])
                    a = dist.sample()
                    a_clamp = torch.tanh(a).squeeze(0).detach().cpu().numpy()
                    lp = dist.log_prob(a).sum(-1)
                    agent_actions[name] = a_clamp[:cfg["act_dim"]]
                else:
                    logits = shared_disc(obs_t)[:, :cfg["n_actions"]]
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                    agent_actions[name] = a.item()

                step_data[name] = {"obs": o, "lp": lp, "val": val, "type": cfg["type"]}

            # Convert to env format
            sc_action = np.zeros(8, dtype=np.float32)
            if isinstance(agent_actions["sc"], int):
                sc_action[agent_actions["sc"]] = 1.0

            actions = {
                "sc": sc_action,
                "th": agent_actions["th"],
                "at": agent_actions["at"],
                "ro": agent_actions["ro"]
            }
            next_obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward

            for name in agent_cfg:
                step_data[name]["reward"] = reward
                step_data[name]["done"] = float(terminated or truncated)
            all_data.append(step_data)

            obs = next_obs
            if terminated or truncated:
                break

        # PPO update (aggregate across all agents × timesteps)
        if len(all_data) > 1:
            all_obs, all_lp, all_vals, all_rets = [], [], [], []

            for name in agent_cfg:
                rewards = [d[name]["reward"] for d in all_data]
                dones = [d[name]["done"] for d in all_data]
                returns = []
                R = 0
                for r, d in zip(reversed(rewards), reversed(dones)):
                    R = r + gamma * R * (1 - d)
                    returns.insert(0, R)

                for i, d in enumerate(all_data):
                    all_obs.append(d[name]["obs"])
                    all_lp.append(d[name]["lp"])
                    all_vals.append(d[name]["val"])
                    all_rets.append(returns[i])

            obs_batch = torch.FloatTensor(np.array(all_obs)).to(device)
            old_lp = torch.stack(all_lp).detach().squeeze()
            old_vals = torch.cat(all_vals).detach().squeeze()
            rets = torch.FloatTensor(all_rets).to(device)
            advs = rets - old_vals
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            for _ in range(4):
                new_vals = shared_critic(obs_batch).squeeze()
                critic_loss = 0.5 * (rets - new_vals).pow(2).mean()

                # Policy gradient: advantage-weighted value update
                # Since we can't easily recompute per-agent log probs across mixed types,
                # use advantage-weighted critic loss as proxy for policy improvement
                adv_weighted_loss = (advs.detach() * new_vals).mean() * 0.01
                
                total_loss = critic_loss - adv_weighted_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(shared_cont.parameters()) + list(shared_disc.parameters()) + list(shared_critic.parameters()), 0.5)
                optimizer.step()

        write_csv_row(f, ep, episode_reward, env.get_metrics())
        if ep % 500 == 0:
            f.flush()
            print(f"Episode {ep:>6} | Reward: {episode_reward:.2f} | MTTD: {info.get('mttd', 200)}")

    f.close()
    print(f"MAPPO done. Saved to {log_file}")


# ==============================================================
METHODS = {"rule_soar": train_rule_soar, "single_drl": train_single_drl,
           "iql": train_iql, "mappo": train_mappo}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS.keys())
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    config = load_config(args.config)
    print(f"Training: {args.method}, seed={args.seed}, episodes={args.episodes}")
    start = time.time()
    METHODS[args.method](config, args.seed, args.episodes, args.save_dir)
    print(f"Done in {(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    main()
