# Configuration Files Guide

## Recommended Configs

### For Best Results (Recommended)
- **`train_ppo_best.yaml`** - Optimal configuration from experiment analysis
  - Lambda: 0.5 (optimal balance)
  - Epochs: 4 (prevents overfitting)
  - Timesteps: 100K
  - Use this for production training

### For Extended Training
- **`train_ppo_optimized.yaml`** - Maximum performance configuration
  - Lambda: 0.8 (stronger penalty)
  - Epochs: 8 (more learning per rollout)
  - Timesteps: 200K
  - Use when you need best possible results

### For Quick Iteration
- **`train_ppo_fast_optimized.yaml`** - Fast training configuration
  - Lambda: 0.7 (balanced)
  - Epochs: 8
  - Timesteps: 30K (~45-90 minutes)
  - Use for rapid prototyping

## Base Configs

- **`base.yaml`** - Base settings shared across experiments
- **`collect_latents.yaml`** - Phase 1: Latent collection
- **`train_probes.yaml`** - Phase 1: Probe training

## Platform-Specific

- **`rtx4050_optimized.yaml`** - Optimized for 6GB VRAM GPUs
- **`colab_optimized.yaml`** - Optimized for Google Colab

## Experiments

- **`experiments/*.yaml`** - Hyperparameter experiment configs
  - See `EXPERIMENTS.md` for details
  - See `EXPERIMENT_ANALYSIS.md` for results

## Usage

```bash
# Train with best config
python scripts/train_ppo.py --config configs/train_ppo_best.yaml

# Train with extended config
python scripts/train_ppo.py --config configs/train_ppo_optimized.yaml

# Quick training
python scripts/train_ppo.py --config configs/train_ppo_fast_optimized.yaml
```

