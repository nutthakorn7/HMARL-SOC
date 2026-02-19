"""Threat Hunter agent using SAC (Tier 2)."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
import copy

from ..models.networks import GaussianActor, TwinQNetwork


class ThreatHunter:
    """
    Tier 2: Threat Hunter using Soft Actor-Critic (SAC).
    Continuous action space for investigation depth/scope.
    Entropy maximization supports exploration for unknown threats.
    """
    
    def __init__(self, obs_dim: int = 128, action_dim: int = 16,
                 hidden_dim: int = 256, lr: float = 3e-4,
                 gamma: float = 0.99, alpha: float = 0.2,
                 tau: float = 0.005, auto_alpha: bool = True,
                 device: torch.device = torch.device("cpu")):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
    
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select continuous hunting action."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                _, _, mean = self.actor.sample(obs_t)
                return mean.cpu().numpy().flatten()
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(obs_t)
                return action.cpu().numpy().flatten()
    
    def update(self, batch: dict, weights: np.ndarray = None) -> dict:
        """SAC update step."""
        obs = batch["obs_th"]
        actions = batch["act_th"]
        rewards = batch["reward"]
        next_obs = batch["next_obs_th"]
        dones = batch["done"]
        
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # ---- Critic update ----
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_target
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---- Actor update ----
        new_actions, log_probs, _ = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ---- Alpha update ----
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # ---- Soft target update ----
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # TD errors for priority update
        td_errors = (torch.abs(q1 - target_q) + torch.abs(q2 - target_q)) / 2
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "td_errors": td_errors.detach().cpu().numpy().flatten(),
        }
