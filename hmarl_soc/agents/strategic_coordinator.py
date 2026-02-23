"""Strategic Coordinator agent using PPO (Tier 1)."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

from ..models.networks import ActorCritic


class StrategicCoordinator:
    """
    Tier 1: Strategic Coordinator using PPO.
    Operates at slower timescale (every K steps).
    Produces directives for operational agents.
    """
    
    def __init__(self, obs_dim: int = 64, action_dim: int = 8,
                 hidden_dim: int = 256, lr: float = 3e-4,
                 gamma: float = 0.99, clip_eps: float = 0.2,
                 entropy_coeff: float = 0.01, K: int = 10,
                 device: torch.device = torch.device("cpu")):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.K = K
        self.device = device
        
        self.network = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO rollout buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.logprob_buffer = []
        self.value_buffer = []
        self.done_buffer = []
    
    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Select directive action."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _ = self.network.get_action(obs_t)
        
        action = action.item()
        log_prob = log_prob.item()
        
        # Store for training
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.logprob_buffer.append(log_prob)
        
        with torch.no_grad():
            _, value = self.network(obs_t)
            self.value_buffer.append(value.item())
        
        return action, log_prob
    
    def store_reward(self, reward: float, done: bool):
        """Store cumulated reward (over K steps)."""
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
    
    def update(self, epochs: int = 4) -> float:
        """PPO update using collected rollout."""
        if len(self.obs_buffer) < 3:
            self.obs_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.logprob_buffer.clear()
            self.value_buffer.clear()
            self.done_buffer.clear()
            return 0.0
        
        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        actions = torch.LongTensor(self.action_buffer).to(self.device)
        old_logprobs = torch.FloatTensor(self.logprob_buffer).to(self.device)
        rewards = self.reward_buffer
        dones = self.done_buffer
        values = self.value_buffer
        
        # Compute returns and advantages (GAE)
        returns = []
        advantages = []
        gae = 0
        lam = 0.95
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_loss = 0
        for _ in range(epochs):
            log_probs, entropy, v_pred = self.network.evaluate(obs, actions)
            
            # Policy loss
            ratio = torch.exp(log_probs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(v_pred.view(-1), returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + 0.5 * value_loss + self.entropy_coeff * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.logprob_buffer.clear()
        self.value_buffer.clear()
        self.done_buffer.clear()
        
        return total_loss / epochs
    
    def get_directive(self, action: int) -> np.ndarray:
        """Convert discrete action to 8-dim strategic directive vector.
        
        Dims: [0:2] TH scan boost, [2:4] AT aggression, [4:6] RO urgency, [6:8] coordination
        Values in [-1, 1] range for env.sc_directive processing.
        """
        directive = np.zeros(8, dtype=np.float32)
        
        priority = action % 4   # 0=low, 1=medium, 2=high, 3=critical
        focus_seg = action // 4  # which segment to focus on (0 or 1)
        
        # Scale priority to [-1, 1]
        p = -1.0 + 2.0 * (priority / 3.0)  # low=-1, critical=1
        
        directive[0] = p          # TH scan intensity boost
        directive[1] = p * 0.8    # TH scope boost
        directive[2] = p          # AT aggression
        directive[3] = float(focus_seg)  # focus segment indicator
        directive[4] = p          # RO urgency
        directive[5] = p * 0.6    # RO remediation priority
        directive[6] = p * 0.5    # coordination strength
        directive[7] = float(focus_seg) * 0.5  # coordination target
        
        return directive
