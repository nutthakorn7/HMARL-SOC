"""Multi-head attention-based explainability module for HMARL-SOC."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class AttentionExplainer(nn.Module):
    """
    Multi-head attention mechanism for agent decision explainability.
    Computes: α_ij = softmax(q_i^T k_j / sqrt(d_k))
    
    Generates attention maps showing which input features most influenced
    each agent's decision, enabling SOC analysts to audit automated actions.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 4, d_k: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.scale = d_k ** 0.5
        
        self.W_q = nn.Linear(feature_dim, num_heads * d_k)
        self.W_k = nn.Linear(feature_dim, num_heads * d_k)
        self.W_v = nn.Linear(feature_dim, num_heads * d_k)
        self.W_out = nn.Linear(num_heads * d_k, feature_dim)
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: agent's current obs/action embedding [batch, feature_dim]
            keys: input feature vectors [batch, seq_len, feature_dim]
            values: if None, uses keys
        
        Returns:
            output: attended features [batch, feature_dim]
            attention_weights: [batch, num_heads, seq_len] for explainability
        """
        if values is None:
            values = keys
        
        batch_size = query.size(0)
        seq_len = keys.size(1)
        
        # Project to multi-head Q, K, V
        Q = self.W_q(query).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(keys).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(values).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, heads, 1, seq]
        attention_weights = F.softmax(scores, dim=-1)  # [batch, heads, 1, seq]
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch, heads, 1, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)
        
        output = self.W_out(context)
        
        # Return weights for explainability [batch, heads, seq_len]
        attn_map = attention_weights.squeeze(2)
        
        return output, attn_map
    
    def explain(self, query: torch.Tensor, keys: torch.Tensor,
                feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Generate human-readable explanation of agent decision.
        
        Returns dict mapping feature names to importance scores.
        """
        with torch.no_grad():
            _, attn_weights = self.forward(query, keys)
        
        # Average across heads
        importance = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        explanation = {name: float(score) for name, score in zip(feature_names, importance)}
        # Sort by importance
        explanation = dict(sorted(explanation.items(), key=lambda x: x[1], reverse=True))
        
        return explanation
