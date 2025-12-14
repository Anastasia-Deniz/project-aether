# PPO Training Experiments

This document describes the hyperparameter experiments set up for comparing different training configurations.

## Experiment Overview

We're testing key hyperparameters that affect PPO training performance:

1. **Lambda Transport** (λ) - Controls penalty for large actions
2. **Learning Rate** - Controls learning speed and stability
3. **N Epochs** - Number of PPO updates per rollout
4. **Policy Capacity** - Network size

## Experiments

### Lambda Transport Experiments

| Experiment | Lambda | Hypothesis |
|------------|--------|------------|
| `exp_lambda_0.3` | 0.3 | Low penalty allows larger actions, may improve SSR but increase transport cost |
| `exp_lambda_0.5` | 0.5 | Baseline - balanced tradeoff |
| `exp_lambda_0.8` | 0.8 | High penalty forces efficient actions (from V2 config) |
| `exp_lambda_1.0` | 1.0 | Very high penalty - may be too restrictive |

**Expected Results:**
- Lower λ → Higher transport cost, potentially better SSR
- Higher λ → Lower transport cost, potentially worse SSR

### Learning Rate Experiments

| Experiment | Learning Rate | Hypothesis |
|------------|---------------|------------|
| `exp_lr_1e4` | 1.0e-4 | Lower LR - more stable but slower learning |
| `exp_lr_2e4` | 2.0e-4 | Higher LR - faster learning but less stable |

**Baseline:** 1.5e-4 (from V2 config)

**Expected Results:**
- Lower LR → More stable, slower convergence
- Higher LR → Faster learning, potential instability

### N Epochs Experiments

| Experiment | N Epochs | Hypothesis |
|------------|----------|------------|
| `exp_epochs_4` | 4 | Fewer updates - prevents overfitting to rollout data |
| `exp_epochs_10` | 10 | More updates - extracts more learning per rollout |

**Baseline:** 8 (from V2 config)

**Expected Results:**
- Fewer epochs → Less overfitting, may need more rollouts
- More epochs → More learning per rollout, risk of overfitting

### Policy Capacity Experiments

| Experiment | Hidden Dims | Hypothesis |
|------------|-------------|------------|
| `exp_policy_small` | [256, 128] | Smaller policy - more stable but less expressive |

**Baseline:** [512, 256] (from V2 config)

**Expected Results:**
- Smaller policy → More stable, less capacity
- Larger policy → More expressive, potentially better performance

## Running Experiments

### Run All Experiments

```bash
python scripts/run_experiments.py
```

### Run Specific Experiments

```bash
python scripts/run_experiments.py --experiments exp_lambda_0.3 exp_lambda_0.5
```

### Skip Completed Experiments

```bash
python scripts/run_experiments.py --skip-completed
```

### List Available Experiments

```bash
python scripts/run_experiments.py --list
```

## Experiment Configuration

All experiments use:
- **Total timesteps:** 100,000 (shorter for faster iteration)
- **N steps:** 64 (rollout length)
- **Batch size:** 8
- **Lambda transport:** Varies by experiment
- **Learning rate:** Varies by experiment
- **N epochs:** Varies by experiment
- **Policy capacity:** Varies by experiment

## Output Structure

Each experiment creates:
```
outputs/ppo/{experiment_name}_{timestamp}/
├── final_policy.pt
├── training_history.json
├── checkpoint_*.pt
└── config.yaml
```

## Comparing Results

After running experiments, compare:

1. **Training curves** - `training_history.json` from each experiment
2. **Final metrics** - Last reward, policy loss, value loss
3. **Evaluation** - Run `evaluate_ppo.py` on each final policy

### Quick Comparison Script

```python
import json
from pathlib import Path

# Load all training histories
results = {}
for exp_dir in Path("outputs/ppo").glob("exp_*"):
    history_file = exp_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            data = json.load(f)
            results[exp_dir.name] = {
                "final_reward": data["rewards"][-1] if data["rewards"] else None,
                "final_policy_loss": data["policy_losses"][-1] if data["policy_losses"] else None,
                "final_value_loss": data["value_losses"][-1] if data["value_losses"] else None,
            }

# Print comparison
for exp, metrics in sorted(results.items()):
    print(f"{exp}:")
    print(f"  Reward: {metrics['final_reward']:.4f}")
    print(f"  Policy Loss: {metrics['final_policy_loss']:.4f}")
    print(f"  Value Loss: {metrics['final_value_loss']:.4f}")
```

## Expected Training Time

- **Per experiment:** ~2-4 hours (100K timesteps on RTX 4050)
- **All experiments:** ~20-30 hours total

## Next Steps After Experiments

1. **Identify best hyperparameters** based on training curves
2. **Run longer training** (200K timesteps) with best config
3. **Evaluate** all policies with `evaluate_ppo.py`
4. **Compare metrics:** SSR, FPR, LPIPS, Transport Cost

## Notes

- Experiments use shorter timesteps (100K) for faster iteration
- All experiments use the same probe (auto-detected)
- Early stopping is disabled for fair comparison
- Results saved to `outputs/experiments_summary.json`

