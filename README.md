# HMARL-SOC

**Hierarchical Multi-Agent Reinforcement Learning for Autonomous SOC Operations**

> Accompanying code for the paper submitted to ITC-CSCC 2026.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutthakorn7/HMARL-SOC/blob/main/train_colab.ipynb)

## Architecture

```
┌─────────────────────────────────────────────┐
│      Strategic Coordinator (PPO)            │  Tier 1
│      Campaign Decomposition & Allocation    │
├──────────┬──────────────┬───────────────────┤
│  Threat  │  Alert       │  Response         │  Tier 2
│  Hunter  │  Triage      │  Orchestrator     │
│  (SAC)   │  (DQN)       │  (MADDPG)         │
├──────────┴──────────────┴───────────────────┤
│  Shared Replay Buffer + Attention Explainer │  Tier 3
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (short run)
python train.py --config configs/default.yaml --episodes 1000 --seed 42

# Full training (paper reproduction)
python train.py --config configs/default.yaml --episodes 500000 --seed 42

# Evaluate
python evaluate.py --checkpoint checkpoints/checkpoint_best.pt --episodes 1000
```

Or use **Google Colab** (free GPU): click the badge above ☝️

## Project Structure

```
├── configs/default.yaml          # Hyperparameters (from paper)
├── hmarl_soc/
│   ├── env/
│   │   ├── soc_env.py            # Gymnasium SOC environment (Dec-POMDP)
│   │   ├── network.py            # Enterprise network graph (5 segments, 200 hosts)
│   │   └── attacker.py           # MITRE ATT&CK multi-stage attacker
│   ├── agents/
│   │   ├── strategic_coordinator.py  # PPO with GAE (Tier 1)
│   │   ├── threat_hunter.py          # SAC with auto-entropy (Tier 2)
│   │   ├── alert_triage.py           # Dueling Double DQN (Tier 2)
│   │   └── response_orchestrator.py  # MADDPG (Tier 2)
│   ├── models/networks.py        # Neural networks (3×256 MLP, ReLU)
│   └── core/
│       ├── replay_buffer.py      # Prioritized shared replay buffer
│       └── attention.py          # Multi-head attention explainer
├── train.py                      # Training script (Algorithm 1)
├── evaluate.py                   # Evaluation & Table I generation
└── train_colab.ipynb             # Google Colab notebook (GPU)
```

## Key Results

| Method | MTTD (steps↓) | MTTR (steps↓) | FPR (%↓) | CSR (%↑) |
|--------|:---:|:---:|:---:|:---:|
| Rule-SOAR | 38.4±2.1 | 52.7±3.4 | 18.3±1.4 | 71.2±3.1 |
| Single-DRL | 28.1±1.8 | 35.4±2.6 | 12.7±0.9 | 79.8±2.5 |
| IQL | 25.6±1.5 | 31.2±2.3 | 11.4±0.8 | 82.5±2.2 |
| MAPPO | 23.3±1.2 | 28.6±1.9 | 9.2±0.6 | 87.3±1.8 |
| **HMARL-SOC** | **20.2±0.9** | **25.2±1.4** | **6.1±0.5** | **94.6±1.2** |

*Mean ± std over 5 random seeds.*

## Hyperparameters

- γ = 0.99, η = 3×10⁻⁴, buffer = 10⁶, batch = 256
- K = 10 (SC temporal abstraction)
- Reward: α=1.0, β=1.5, δ=-0.3, λ=-2.0
- Networks: 3-layer MLP, 256 hidden, ReLU
- Complexity: O(N·d²) per step (linear in number of agents)

## Citation

```bibtex
@inproceedings{hmarl-soc2026,
  title={HMARL-SOC: Hierarchical Multi-Agent Reinforcement Learning
         for Autonomous SOC Operations},
  author={Anonymous},
  booktitle={Proc. ITC-CSCC},
  year={2026}
}
```

## License

MIT
