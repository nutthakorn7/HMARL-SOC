"""Neural network architectures for HMARL-SOC agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class MLP(nn.Module):
    """3-layer MLP with 256 hidden units and ReLU (paper spec)."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: int = 256, num_layers: int = 3,
                 output_activation: Optional[str] = None):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "softmax":
            layers.append(nn.Softmax(dim=-1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO (Strategic Coordinator)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = MLP(obs_dim, action_dim, hidden_dim)
        self.critic = MLP(obs_dim, 1, hidden_dim)
    
    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        return probs, self.critic(obs)
    
    def get_action(self, obs: torch.Tensor):
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()
    
    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action), dist.entropy(), self.critic(obs)


class GaussianActor(nn.Module):
    """Gaussian actor for SAC (Threat Hunter) — continuous actions."""
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = MLP(obs_dim, hidden_dim, hidden_dim, num_layers=2)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        # Log prob with tanh correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, mean


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, 1, hidden_dim)
        self.q2 = MLP(obs_dim + action_dim, 1, hidden_dim)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class DQNetwork(nn.Module):
    """Dueling DQN for Alert Triage — discrete actions."""
    
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.feature = MLP(obs_dim, hidden_dim, hidden_dim, num_layers=2)
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature(obs)
        value = self.value(features)
        advantage = self.advantage(features)
        # Dueling architecture
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DDPGActor(nn.Module):
    """Deterministic actor for MADDPG (Response Orchestrator)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_dim, output_activation="tanh")
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DDPGCritic(nn.Module):
    """Centralized critic for MADDPG."""
    
    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(total_obs_dim + total_action_dim, 1, hidden_dim)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)

