"""Alert Triage agent using DQN (Tier 2)."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
import copy

from ..models.networks import DQNetwork


class AlertTriage:
    """
    Tier 2: Alert Triage using Deep Q-Network (DQN).
    Discrete actions: escalate, suppress, correlate, enrich.
    Uses dueling architecture + double DQN.
    """
    
    def __init__(self, obs_dim: int = 64, num_actions: int = 4,
                 hidden_dim: int = 256, lr: float = 3e-4,
                 gamma: float = 0.99, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_decay: int = 50000,
                 target_update: int = 1000,
                 device: torch.device = torch.device("cpu")):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.device = device
        
        self.q_net = DQNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.step_count = 0
    
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        """Select discrete triage action with epsilon-greedy."""
        self.step_count += 1
        
        if not evaluate:
            eps = self.eps_end + (self.eps_start - self.eps_end) * \
                  np.exp(-self.step_count / self.eps_decay)
            if np.random.random() < eps:
                return np.random.randint(self.num_actions)
        
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
            return q_values.argmax(dim=1).item()
    
    def update(self, batch: dict, weights: np.ndarray = None) -> dict:
        """Double DQN update."""
        obs = batch["obs_at"]
        actions = batch["act_at"].long()
        rewards = batch["reward"]
        next_obs = batch["next_obs_at"]
        dones = batch["done"]
        
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Current Q values
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss with optional importance weights
        td_error = torch.abs(q_values - target_q)
        if weights is not None:
            weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
            loss = (weights_t * nn.MSELoss(reduction='none')(q_values, target_q)).mean()
        else:
            loss = nn.MSELoss()(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Target network update
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return {
            "loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "td_errors": td_error.detach().cpu().numpy().flatten(),
        }
