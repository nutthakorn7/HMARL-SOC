# HMARL-SOC

**Hierarchical Multi-Agent Reinforcement Learning for Autonomous Threat Hunting and Automated Incident Response in Enterprise Security Operations Centers**

> Accompanying code for the paper submitted to ITC-CSCC 2026.

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
# Install dependencies
pip install -r requirements.txt

# Train (short run for testing)
python train.py --config configs/default.yaml --episodes 1000 --seed 42

# Full training (paper reproduction)
python train.py --config configs/default.yaml --episodes 500000 --seed 42

# Evaluate
python evaluate.py --checkpoint checkpoints/checkpoint_best.pt --episodes 1000
```

## Project Structure

```
├── configs/default.yaml          # Hyperparameters (from paper)
├── hmarl_soc/
│   ├── env/
│   │   ├── soc_env.py            # Gymnasium SOC environment
│   │   ├── network.py            # Enterprise network graph
│   │   └── attacker.py           # MITRE ATT&CK attacker
│   ├── agents/
│   │   ├── strategic_coordinator.py  # PPO (Tier 1)
│   │   ├── threat_hunter.py          # SAC (Tier 2)
│   │   ├── alert_triage.py           # DQN (Tier 2)
│   │   └── response_orchestrator.py  # MADDPG (Tier 2)
│   ├── models/networks.py        # Neural networks (3×256 MLP)
│   └── core/
│       ├── replay_buffer.py      # Prioritized shared buffer
│       └── attention.py          # Multi-head attention explainer
├── train.py                      # Training script (Algorithm 1)
└── evaluate.py                   # Evaluation & results
```

## Key Results

| Method | MTTD (steps↓) | MTTR (steps↓) | FPR (%↓) | CSR (%↑) |
|--------|:---:|:---:|:---:|:---:|
| Rule-SOAR | 38.4 | 52.7 | 18.3 | 71.2 |
| Single-DRL | 28.1 | 35.4 | 12.7 | 79.8 |
| IQL | 25.6 | 31.2 | 11.4 | 82.5 |
| MAPPO | 23.3 | 28.6 | 9.2 | 87.3 |
| **HMARL-SOC** | **20.2** | **25.2** | **6.1** | **94.6** |

## Hyperparameters

- γ = 0.99, η = 3×10⁻⁴, buffer = 10⁶, batch = 256
- K = 10 (SC temporal abstraction)
- Reward: α=1.0, β=1.5, δ=-0.3, λ=-2.0
- Networks: 3-layer MLP, 256 hidden, ReLU

## Citation

```bibtex
@inproceedings{hmarl-soc2026,
  title={HMARL-SOC: A Hierarchical Multi-Agent Reinforcement Learning Framework
         for Autonomous Threat Hunting and Automated Incident Response
         in Enterprise Security Operations Centers},
  author={Anonymous},
  booktitle={Proc. ITC-CSCC},
  year={2026}
}
```

## License

MIT
