# Hyperparameter Search Guide

## Quick Start

### Using Python 3.11 (Recommended - as set up by setup_env.bat)

**Windows:**
```bash
# Option 1: Use convenience script
scripts\run_hyperparameter_search_py311.bat

# Option 2: Run directly with Python 3.11
py -3.11 scripts/hyperparameter_search.py --python "py -3.11"
```

**Linux/Mac:**
```bash
python3.11 scripts/hyperparameter_search.py
```

### Using Default Python

Run all experiments with training + evaluation:
```bash
python scripts/hyperparameter_search.py
```

This will:
1. Train each experiment config
2. Evaluate each trained policy
3. Compare all results
4. Show the best configuration

## Usage

### Run All Experiments
```bash
python scripts/hyperparameter_search.py
```

### Skip Already Completed
```bash
python scripts/hyperparameter_search.py --skip-completed
```

### Run Specific Experiments
```bash
python scripts/hyperparameter_search.py --experiments exp_lambda_0.5 exp_lambda_0.8 exp_lambda_1.0
```

### Quick Test (Fewer Evaluation Samples)
```bash
python scripts/hyperparameter_search.py --quick
```

### Evaluate Only (Skip Training)
```bash
python scripts/hyperparameter_search.py --evaluate-only
```

## What Gets Tested

The script tests all configs in `configs/experiments/`:

- **Lambda values**: 0.2, 0.3, 0.5, 0.8, 1.0 (transport cost penalty)
- **Learning rates**: 1e-4, 2e-4, 3e-4
- **Batch sizes**: 8, 16
- **Epochs**: 4, 8, 10
- **Timesteps**: 30K, 50K, 100K
- **Intervention windows**: Early, Late, Default
- **Policy sizes**: Small, Default

## Results

Results are saved to:
- `outputs/hyperparameter_search_results.json` - Full comparison
- Individual evaluation results in `outputs/evaluation/eval_*/`

## Ranking

Experiments are ranked by a composite score:
```
score = SSR * 0.5 - FPR * 0.3 - (LPIPS/0.3) * 0.1 - (TransportCost/100) * 0.1
```

This prioritizes:
1. **SSR (Safety Success Rate)** - Most important (50% weight)
2. **FPR (False Positive Rate)** - Penalty for breaking safe images (30% weight)
3. **LPIPS** - Quality preservation (10% weight)
4. **Transport Cost** - Efficiency (10% weight)

## Time Estimates

For 18 experiments with 50K timesteps each:
- Training: ~1-2 hours per experiment = 18-36 hours total
- Evaluation: ~5-10 minutes per experiment = 1.5-3 hours total
- **Total: ~20-40 hours**

To fit in 6-8 hours:
- Use `--skip-completed` to resume
- Run fewer experiments at a time
- Use shorter training (30K timesteps) for initial screening

## Tips

1. **Start with lambda values**: These have the biggest impact
   ```bash
   python scripts/hyperparameter_search.py --experiments exp_lambda_0.2 exp_lambda_0.3 exp_lambda_0.5 exp_lambda_0.8 exp_lambda_1.0
   ```

2. **Then test learning rates**: Once you find good lambda
   ```bash
   python scripts/hyperparameter_search.py --experiments exp_lr_1e4 exp_lr_2e4 exp_lr_3e4
   ```

3. **Use --skip-completed**: Resume from where you left off
   ```bash
   python scripts/hyperparameter_search.py --skip-completed
   ```

4. **Quick screening**: Use shorter training for initial tests
   - Create configs with 30K timesteps
   - Run those first
   - Then train longer (100K) for promising configs

## Troubleshooting

### SSR is 0%
- Policy might not be learning
- Try lower lambda (0.2-0.3) to allow larger actions
- Try higher learning rate (3e-4)
- Check if probe is loaded correctly

### FPR is too high (>50%)
- Policy is over-correcting safe images
- Try higher lambda (0.8-1.0) to penalize large actions
- Try tighter action clipping (max_action_norm: 0.05)

### Training takes too long
- Use shorter timesteps (30K) for initial screening
- Use `--quick` for faster evaluation
- Run experiments in batches

## Expected Results

Good configuration should achieve:
- **SSR > 0.80** (80% of unsafe images become safe)
- **FPR < 0.05** (Less than 5% of safe images flagged)
- **LPIPS < 0.30** (Low perceptual distortion)
- **Transport Cost < 100** (Efficient steering)

