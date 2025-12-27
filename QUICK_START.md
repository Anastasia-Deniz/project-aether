# Quick Start Guide - Running Experiments

## The Issue

You have multiple Python installations. The `setup_env.bat` uses `py -3.11` (where packages are installed), but scripts default to `python` (different interpreter).

## Solution

Always use `py -3.11` instead of `python`:

### Option 1: Use the Fixed Scripts

I've updated the scripts to automatically use `py -3.11`. Just run:

```powershell
python run_experiments_complete.py --skip_phase1
```

### Option 2: Run Commands Directly with py -3.11

```powershell
# Phase 2: Train PPO
py -3.11 scripts/train_ppo.py --config configs/train_ppo_best.yaml --probe_path checkpoints/probes/run_20251225_183438/pytorch/

# Phase 3: Evaluate (Robust - Recommended)
py -3.11 scripts/evaluate_ppo_robust.py --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt --probe_path checkpoints/probes/run_20251225_183438/pytorch/ --num_samples 100 --seed 42
```

### Option 3: Use start_training.py (Fixed)

```powershell
py -3.11 start_training.py
```

## Verify Setup

```powershell
py -3.11 scripts/test_setup.py
```

## Why This Happens

- `py -3.11` → Python 3.11 (where packages are installed) ✅
- `python` → Python 3.14 (different installation, no packages) ❌

The scripts now automatically detect and use `py -3.11` if available.

