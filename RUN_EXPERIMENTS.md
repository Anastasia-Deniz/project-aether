# Running Project Aether Experiments

## Quick Start

Since you have existing probes and latents, you can start directly with Phase 2 training:

### Option 1: Use the Complete Runner Script

```powershell
# Run everything (uses existing probes)
python run_experiments_complete.py --skip_phase1

# Or run from scratch
python run_experiments_complete.py --full
```

### Option 2: Run Individual Phases

#### Phase 2: Train PPO Policy

```powershell
python scripts/train_ppo.py --config configs/train_ppo_best.yaml --probe_path checkpoints/probes/run_20251225_183438/pytorch/
```

#### Phase 3: Evaluate Policy (Robust - Recommended)

```powershell
python scripts/evaluate_ppo_robust.py --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt --probe_path checkpoints/probes/run_20251225_183438/pytorch/ --num_samples 100 --seed 42
```

## Current Status

✅ **Phase 1 Complete**: 
- Latest probe run: `run_20251225_183438`
- Best accuracy: 85% at timestep 13
- Optimal intervention window: [10, 14]
- Sensitivity scores updated with metadata

⏳ **Phase 2**: Ready to start training

⏳ **Phase 3**: Will run after Phase 2 completes

## Configuration

The training uses `configs/train_ppo_best.yaml` which has:
- Lambda = 0.5 (optimal from experiments)
- 100K timesteps
- 4 epochs (prevents overfitting)
- Learning rate = 1.5e-4

## Expected Training Time

- Phase 2: ~4-6 hours on GPU (RTX 4050/3060)
- Phase 3: ~10-20 minutes

## Monitoring

Training progress will be logged to:
- Console output
- `outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/training_history.json`

