---
description: How to run HMARL-SOC training (optimized)
---

# HMARL-SOC Training

> **ALWAYS use `train_fast.py`** — never use `train.py` (deprecated).

// turbo-all

1. Activate the Python 3.12 venv:
```bash
source ~/.venvs/hmarl/bin/activate
```

2. Run single seed:
```bash
cd "/Users/pop7/Library/CloudStorage/OneDrive-MonsterConnectCo.,Ltd/Research/Submited_CF3 HMARL-SOC/code"
python3 train_fast.py --config configs/default.yaml --episodes 10000 --seed 42 --workers 4
# Add --fast-buffer ONLY for debug/testing (worse results but ~10% faster)
```

3. Run all seeds (2 in parallel):
```bash
bash train_multi_seed_fast.sh 42 123 456 789 1024
```

## Key Info
- **Venv**: `~/.venvs/hmarl` (Python 3.12.12, PyTorch 2.10.0)
- **Optimizations**: vectorized envs, torch.compile, mixed precision, uniform buffer
- **Speed**: ~7x faster than legacy `train.py`
- **Output**: Same CSV + checkpoint format, fully compatible with `evaluate.py`
