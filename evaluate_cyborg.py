#!/usr/bin/env python3
"""
CybORG CAGE-4 Evaluation for HMARL-SOC
Evaluates trained HMARL-SOC agents on the CAGE Challenge 4 environment
via zero-shot transfer with observation/action space adapters.

Maps HMARL-SOC hierarchy to CybORG blue agents:
  SC  (PPO)    → blue_agent_4 (HQ coordinator, 3 subnets)
  TH  (SAC)    → blue_agent_0, blue_agent_1 (scanning/detection)
  AT  (DQN)    → blue_agent_2 (alert triage)
  RO  (MADDPG) → blue_agent_3 (response)
"""

import argparse
import os
import yaml
import numpy as np
import torch
from collections import defaultdict

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.Wrappers import BlueEnterpriseWrapper

from hmarl_soc.agents.strategic_coordinator import StrategicCoordinator
from hmarl_soc.agents.threat_hunter import ThreatHunter
from hmarl_soc.agents.alert_triage import AlertTriage
from hmarl_soc.agents.response_orchestrator import ResponseOrchestrator


class CybORGAdapter:
    """Adapts CybORG CAGE-4 observations/actions to HMARL-SOC format."""
    
    # CybORG obs dimensions
    SUBNET_OBS_DIM = 92    # blue_agent 0-3
    HQ_OBS_DIM = 210       # blue_agent 4
    
    # HMARL-SOC obs dimensions (from config)
    SC_OBS_DIM = 64
    TH_OBS_DIM = 128
    AT_OBS_DIM = 64
    RO_OBS_DIM = 96
    
    def __init__(self, device="cpu"):
        self.device = device
        # Learnable projection layers (random but fixed for consistency)
        torch.manual_seed(42)
        # Project CybORG obs → HMARL-SOC obs dimensions
        self.proj_sc = torch.nn.Linear(self.HQ_OBS_DIM, self.SC_OBS_DIM).to(device)
        self.proj_th = torch.nn.Linear(self.SUBNET_OBS_DIM, self.TH_OBS_DIM).to(device)
        self.proj_at = torch.nn.Linear(self.SUBNET_OBS_DIM, self.AT_OBS_DIM).to(device)
        self.proj_ro = torch.nn.Linear(self.SUBNET_OBS_DIM, self.RO_OBS_DIM).to(device)
        
        # Freeze projections
        for proj in [self.proj_sc, self.proj_th, self.proj_at, self.proj_ro]:
            for p in proj.parameters():
                p.requires_grad = False
    
    def adapt_obs(self, cyborg_obs: dict) -> dict:
        """Convert CybORG observations to HMARL-SOC format."""
        with torch.no_grad():
            # SC gets blue_agent_4 (HQ) observation
            sc_raw = torch.FloatTensor(cyborg_obs["blue_agent_4"]).to(self.device)
            sc_obs = self.proj_sc(sc_raw).cpu().numpy()
            
            # TH gets blue_agent_0 + blue_agent_1 (concatenated then projected)
            th_raw = torch.FloatTensor(
                np.concatenate([cyborg_obs["blue_agent_0"], 
                               cyborg_obs["blue_agent_1"][:36]])  # pad to 128
            ).to(self.device)
            th_obs = self.proj_th(torch.FloatTensor(cyborg_obs["blue_agent_0"]).to(self.device)).cpu().numpy()
            
            # AT gets blue_agent_2
            at_raw = torch.FloatTensor(cyborg_obs["blue_agent_2"]).to(self.device)
            at_obs = self.proj_at(at_raw).cpu().numpy()
            
            # RO gets blue_agent_3
            ro_raw = torch.FloatTensor(cyborg_obs["blue_agent_3"]).to(self.device)
            ro_obs = self.proj_ro(ro_raw).cpu().numpy()
        
        return {"sc": sc_obs, "th": th_obs, "at": at_obs, "ro": ro_obs}
    
    def adapt_actions(self, hmarl_actions: dict, env: BlueEnterpriseWrapper) -> dict:
        """Convert HMARL-SOC actions to CybORG discrete actions."""
        cyborg_actions = {}
        
        # SC directive → blue_agent_4 action (map to BlockTraffic/AllowTraffic range)
        sc_directive = hmarl_actions.get("sc")
        if sc_directive is not None:
            sc_action_idx = int(sc_directive) if isinstance(sc_directive, (int, np.integer)) else 0
            # Map to valid range for agent_4 (0-241)
            cyborg_actions["blue_agent_4"] = sc_action_idx % 242
        else:
            cyborg_actions["blue_agent_4"] = 0  # Monitor
        
        # TH action → blue_agent_0 (Analyse actions, indices 0-9)
        th_action = hmarl_actions.get("th")
        if th_action is not None:
            th_idx = int(np.argmax(th_action)) if hasattr(th_action, '__len__') else int(th_action)
            cyborg_actions["blue_agent_0"] = th_idx % 82
        else:
            cyborg_actions["blue_agent_0"] = 0
        
        # TH also controls agent_1
        cyborg_actions["blue_agent_1"] = (cyborg_actions["blue_agent_0"] + 10) % 82
        
        # AT action → blue_agent_2 (map DQN discrete action)
        at_action = hmarl_actions.get("at")
        if at_action is not None:
            at_idx = int(at_action[0]) if hasattr(at_action, '__len__') else int(at_action)
            # Map 4 HMARL-SOC actions to CybORG action ranges
            # 0=Analyse(0-9), 1=Remove(10-25), 2=Restore(26-41), 3=DeployDecoy(42-57)
            action_ranges = [(0, 9), (10, 25), (26, 41), (42, 57)]
            at_idx = at_idx % 4
            start, end = action_ranges[at_idx]
            cyborg_actions["blue_agent_2"] = np.random.randint(start, end + 1)
        else:
            cyborg_actions["blue_agent_2"] = 0
        
        # RO action → blue_agent_3 (Response actions: Remove, Restore, Block)
        ro_action = hmarl_actions.get("ro")
        if ro_action is not None:
            ro_idx = int(np.argmax(ro_action)) if hasattr(ro_action, '__len__') else int(ro_action)
            cyborg_actions["blue_agent_3"] = ro_idx % 82
        else:
            cyborg_actions["blue_agent_3"] = 0
        
        return cyborg_actions


def load_agents(config, checkpoint_path, device):
    """Load pretrained HMARL-SOC agents from checkpoint."""
    sc_cfg = config["agents"]["strategic_coordinator"]
    th_cfg = config["agents"]["threat_hunter"]
    at_cfg = config["agents"]["alert_triage"]
    ro_cfg = config["agents"]["response_orchestrator"]
    gamma = config["training"]["gamma"]
    K = sc_cfg.get("temporal_abstraction_K", 10)
    
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
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sc.network.load_state_dict(ckpt["sc_state"])
        th.actor.load_state_dict(ckpt["th_actor"])
        th.critic.load_state_dict(ckpt["th_critic"])
        at.q_net.load_state_dict(ckpt["at_qnet"])
        ro.actor.load_state_dict(ckpt["ro_actor"])
        ro.critic.load_state_dict(ckpt["ro_critic"])
        print(f"Loaded checkpoint: {checkpoint_path} (episode {ckpt['episode']})")
    else:
        print("No checkpoint — using random initialization")
    
    return sc, th, at, ro


def evaluate_cyborg(config, checkpoint_path, num_episodes=100, seed=42):
    """Evaluate HMARL-SOC on CybORG CAGE-4."""
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"Device: {device}")
    
    # Load agents
    sc, th, at, ro = load_agents(config, checkpoint_path, device)
    adapter = CybORGAdapter(device=device)
    
    # Create CybORG environment
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500
    )
    
    metrics = defaultdict(list)
    K = config["agents"]["strategic_coordinator"].get("temporal_abstraction_K", 10)
    
    print(f"\nEvaluating {num_episodes} episodes on CybORG CAGE-4...")
    print("=" * 60)
    
    for ep in range(1, num_episodes + 1):
        cyborg = CybORG(scenario_generator=sg, seed=seed + ep)
        env = BlueEnterpriseWrapper(env=cyborg)
        cyborg_obs, _ = env.reset()
        
        ep_reward = 0.0
        detections = 0
        responses = 0
        directive = None
        
        for t in range(500):
            # Adapt observations
            hmarl_obs = adapter.adapt_obs(cyborg_obs)
            
            # SC decides every K steps
            if t % K == 0:
                sc_action, _ = sc.select_action(hmarl_obs["sc"])
                directive = sc.get_directive(sc_action)
            
            # Operational agents act
            th_action = th.select_action(hmarl_obs["th"])
            at_action = at.select_action(hmarl_obs["at"])
            ro_action = ro.select_action(hmarl_obs["ro"])
            
            hmarl_actions = {
                "sc": directive,
                "th": th_action,
                "at": at_action,
                "ro": ro_action,
            }
            
            # Adapt to CybORG actions
            cyborg_actions = adapter.adapt_actions(hmarl_actions, env)
            
            # Step
            cyborg_obs, rewards, terms, truncs, infos = env.step(cyborg_actions)
            
            step_reward = sum(rewards.values())
            ep_reward += step_reward
            
            # Track detections (negative reward = red activity detected)
            if step_reward < 0:
                detections += 1
            
            # Check if any agent terminated
            if any(terms.values()) or any(truncs.values()):
                break
        
        metrics["reward"].append(ep_reward)
        metrics["detections"].append(detections)
        metrics["steps"].append(t + 1)
        
        if ep % 10 == 0:
            avg_reward = np.mean(metrics["reward"][-10:])
            avg_det = np.mean(metrics["detections"][-10:])
            print(f"Episode {ep:4d} | Avg Reward: {avg_reward:8.2f} | "
                  f"Avg Detections: {avg_det:.1f} | Steps: {t+1}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CybORG CAGE-4 Evaluation Results")
    print("=" * 60)
    results = {
        "mean_reward": float(np.mean(metrics["reward"])),
        "std_reward": float(np.std(metrics["reward"])),
        "mean_detections": float(np.mean(metrics["detections"])),
        "mean_steps": float(np.mean(metrics["steps"])),
    }
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="HMARL-SOC CybORG Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    results = evaluate_cyborg(config, args.checkpoint, args.episodes, args.seed)
    
    # Save results
    import json
    with open("cyborg_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to cyborg_results.json")


if __name__ == "__main__":
    main()
