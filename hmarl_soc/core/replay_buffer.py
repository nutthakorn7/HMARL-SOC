"""Prioritized shared experience replay buffer for HMARL-SOC."""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class PrioritizedReplayBuffer:
    """
    Shared prioritized experience replay buffer (Schaul et al., 2016).
    Priority = detection_confidence × response_urgency.
    Shared across all agents for cross-agent knowledge transfer.
    """
    
    def __init__(self, capacity: int = 1_000_000, 
                 alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 1e-5):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance sampling exponent
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, transition: dict, priority: Optional[float] = None):
        """
        Store transition with priority.
        
        transition: {
            'obs_sc', 'obs_th', 'obs_at', 'obs_ro',
            'act_sc', 'act_th', 'act_at', 'act_ro',
            'reward', 'done',
            'next_obs_sc', 'next_obs_th', 'next_obs_at', 'next_obs_ro'
        }
        """
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tuple[dict, np.ndarray, np.ndarray]:
        """Sample batch with prioritized probabilities."""
        if self.size == 0:
            raise ValueError("Buffer is empty")
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False if self.size >= batch_size else True)
        
        # Importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Collate batch
        batch = self._collate([self.buffer[i] for i in indices], device)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
    
    def _collate(self, transitions: list, device: torch.device) -> dict:
        """Collate list of transitions into tensors."""
        batch = {}
        keys = transitions[0].keys()
        
        for key in keys:
            values = [t[key] for t in transitions]
            if isinstance(values[0], np.ndarray):
                batch[key] = torch.FloatTensor(np.stack(values)).to(device)
            elif isinstance(values[0], (int, float, bool)):
                batch[key] = torch.FloatTensor(values).to(device)
            else:
                batch[key] = values
        
        return batch
    
    def __len__(self) -> int:
        return self.size
