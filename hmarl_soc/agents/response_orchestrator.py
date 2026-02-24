"""Response Orchestrator agent using MADDPG (Tier 2)."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
import copy

from ..models.networks import DDPGActor, DDPGCritic


class ResponseOrchestrator:
    """
    Tier 2: Response Orchestrator using Multi-Agent DDPG (MADDPG).
    Continuous actions for containment/remediation across network segments.
    Centralized critic with decentralized actor.
    """
    
    def __init__(self, obs_dim: int = 96, action_dim: int = 12,
                 total_obs_dim: int = 352, total_action_dim: int = 37,  # SC:8 + TH:16 + AT:1 + RO:12
                 hidden_dim: int = 256, lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, noise_std: float = 0.1,
                 device: torch.device = torch.device("cpu")):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.device = device
        
        # Actor (decentralized)
        self.actor = DDPGActor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        
        # Critic (centralized — sees all agents' obs and actions)
        self.critic = DDPGCritic(total_obs_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select continuous response action."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        
        if not evaluate:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def update(self, batch: dict, all_obs: torch.Tensor, 
               all_actions: torch.Tensor, all_next_obs: torch.Tensor,
               weights: np.ndarray = None) -> dict:
        """MADDPG update with centralized critic."""
        obs = batch["obs_ro"]
        actions = batch["act_ro"]
        rewards = batch["reward"]
        next_obs = batch["next_obs_ro"]
        dones = batch["done"]
        
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # ---- Critic update ----
        with torch.no_grad():
            # Target actions from target actor
            next_action_ro = self.actor_target(next_obs)
            # Use target actions for next state Q-value
            # Substitute RO's target actions into the combined action tensor
            all_next_actions = all_actions.clone()
            # RO actions are at the end (after SC 8 + TH 16 + AT 1 = 25)
            all_next_actions[:, -self.action_dim:] = next_action_ro
            target_q = self.critic_target(all_next_obs, all_next_actions)
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(all_obs, all_actions)
        critic_loss = nn.MSELoss()(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # ---- Actor update ----
        new_actions = self.actor(obs)
        # Replace RO actions in combined
        all_actions_new = all_actions.clone()
        actor_loss = -self.critic(all_obs, all_actions_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # ---- Soft target update ----
        for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        
        td_errors = torch.abs(current_q - target_value)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "td_errors": td_errors.detach().cpu().numpy().flatten(),
        }
