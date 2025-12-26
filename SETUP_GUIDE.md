# Project Aether - Setup and Run Guide

## Quick Start Commands

### 1. Environment Setup

**Windows PowerShell:**
```powershell
# Activate Python 3.11 environment (if using venv)
.\venv\Scripts\Activate.ps1

# Or use conda
conda activate aether

# Install dependencies
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
py -3.11 -m pip install -r requirements.txt
```

**Or use the setup script:**
```powershell
.\setup_env.ps1
.\setup_env.bat
```

### 2. Verify Installation

```powershell
py -3.11 scripts/test_setup.py
```

**Expected output:**
- ✓ PyTorch
- ✓ CUDA available (or ⚠ CPU mode)
- ✓ All structure checks passed

### 3. Run Unit Tests

```powershell
# Test environment components
py -3.11 tests/test_env.py

# Test reward modules
py -3.11 tests/test_rewards.py

# Test PPO components
py -3.11 tests/test_ppo.py
```

### 4. Phase 1: Linear Probing & Sensitivity Analysis

**Important Model Selection:**
- **Recommended Model:** `CompVis/stable-diffusion-v1-4` (default)
  - Less censored than SD 1.5, better for research on safety/alignment
  - Can generate nudity content for research purposes
  - Well-documented and stable
- **Alternative Models:** SD 1.5 variants are more censored and may not generate the content needed for this research

**Quick test (20 samples, 10 steps):**
```powershell
py -3.11 scripts/run_phase1.py --quick
```

**Full run (50 samples, 20 steps) - Nudity Focus:**
```powershell
# Recommended: Focus on nudity only with strict quality thresholds
py -3.11 scripts/collect_latents.py `
    --num_samples 50 `
    --num_steps 20 `
    --device cuda `
    --focus_nudity `
    --hard_only `
    --min_inappropriate_pct 60.0 `
    --min_nudity_pct 50.0 `
    --model_id CompVis/stable-diffusion-v1-4

# Or use defaults (already configured for nudity focus)
py -3.11 scripts/collect_latents.py --num_samples 50 --num_steps 20 --device cuda
```

**Note:** The project now focuses **only on nudity** (not gore/violence) for clearer concept boundaries and better probe training.

2. Train probes (with improved heuristics):
```powershell
py -3.11 scripts/train_probes.py --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS
```

2.5. (Optional) Measure empirical layer sensitivity ⭐ NEW:
```powershell
# This measures FID and SSR empirically for better accuracy
# Takes ~30-60 minutes but provides more accurate sensitivity scores
py -3.11 scripts/measure_layer_sensitivity.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --num_samples 20 `
    --device cuda

# Then re-train probes with empirical measurements:
py -3.11 scripts/train_probes.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --use_empirical
```

**Note:** Empirical measurement is optional but recommended for best results. It measures:
- **Quality preservation**: FID between steered and unsteered images (Heusel et al., 2017)
- **Steering effectiveness**: SSR improvement from steering

3. Sensitivity analysis (now included in train_probes.py):
The sensitivity analysis is automatically generated during probe training. Check:
- `checkpoints/probes/run_YYYYMMDD_HHMMSS/probe_analysis.png` - Visualization
- `checkpoints/probes/run_YYYYMMDD_HHMMSS/sensitivity_scores.json` - Scores and optimal window

**Output locations:**
- Latents: `data/latents/run_YYYYMMDD_HHMMSS/`
- Probes: `checkpoints/probes/run_YYYYMMDD_HHMMSS/`
- Images: `data/latents/run_YYYYMMDD_HHMMSS/images/` (if --save_images used)

### 4.5. Visualize Generated Images & Verify Probe Accuracy ⭐ NEW

**Important:** After Phase 1, verify what images were actually generated and check if probe accuracy is meaningful.

#### Option A: Generate Images from Collected Latents

If you already collected latents, decode them to see the actual images:

```powershell
# Generate images from final timestep (actual generated images)
py -3.11 scripts/generate_images_from_latents.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --timestep 8 `
    --num_samples 50 `
    --device cuda
```

**Output:**
- Safe images: `data/latents/run_YYYYMMDD_HHMMSS/images_t08/safe/`
- Unsafe images: `data/latents/run_YYYYMMDD_HHMMSS/images_t08/unsafe/`
- HTML viewer: `data/latents/run_YYYYMMDD_HHMMSS/images_t08/viewer.html`

**Open the HTML file in your browser** to see all images side-by-side!

#### Option B: Visualize Probe Results with Predictions

To see probe predictions on images and verify accuracy:

```powershell
# Visualize probe results (shows correct/incorrect classifications)
py -3.11 scripts/visualize_probe_results.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --probe_dir ./checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ `
    --timestep 4 `
    --num_samples 20 `
    --device cuda
```

**Output:**
- Visualization: `outputs/visualizations/probe_visualization_t04.png`
- Detailed results: `outputs/visualizations/probe_results_t04.json`

**What to check:**
- ✓ Are safe images actually safe?
- ✓ Are unsafe images actually unsafe?
- ✓ Does probe accuracy match visual inspection?
- ✓ Check confusion matrix and probability distributions

**Note:** Use the best timestep from sensitivity analysis (usually t=2, t=3, or t=4). Check `checkpoints/probes/run_YYYYMMDD_HHMMSS/sensitivity_scores.json` for the optimal timestep.

#### Quick Verification Workflow

```powershell
# 1. Collect latents (images saved automatically)
py -3.11 scripts/collect_latents.py --num_samples 50 --num_steps 8 --device cuda

# Note the output: data/latents/run_YYYYMMDD_HHMMSS/

# 2. Train probes
py -3.11 scripts/train_probes.py --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS

# Note the output: checkpoints/probes/run_YYYYMMDD_HHMMSS/

# 3. Generate images (if not already saved)
py -3.11 scripts/generate_images_from_latents.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --timestep 8

# 4. Open HTML viewer to see all images
# Open: data/latents/run_YYYYMMDD_HHMMSS/images_t08/viewer.html

# 5. Visualize probe results
py -3.11 scripts/visualize_probe_results.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --probe_dir ./checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ `
    --timestep 4

# 6. Check the visualization PNG and verify probe accuracy makes sense
```

### 5. Phase 2: PPO Policy Training

**Using best configuration (recommended):**
```powershell
py -3.11 scripts/train_ppo.py --config configs/train_ppo_best.yaml
```

**Fast training for Colab (2-3 hours instead of 8 hours):**
```powershell
# Use fast config optimized for Colab T4 GPU
py -3.11 scripts/train_ppo.py --config configs/colab_fast_20steps.yaml
```

**Quick test (minimal settings):**
```powershell
py -3.11 scripts/train_ppo.py --quick
```

**Custom settings:**
```powershell
py -3.11 scripts/train_ppo.py `
    --config configs/train_ppo_best.yaml `
    --total_timesteps 100000 `
    --lambda_transport 0.5 `
    --device cuda
```

**Output location:**
- Policy: `outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt`
- Training history: `outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/training_history.json`

#### Fast Training Configurations

For Colab or time-constrained training, use optimized fast configs:

| Config | Timesteps | Time | Use Case |
|--------|-----------|------|----------|
| `colab_fast_20steps.yaml` | 50K | 2-3h | **Recommended** - Compatible with existing probes |
| `colab_fast.yaml` | 50K | 1.5-2h | Requires probes trained with 8 steps |
| `train_ppo_best.yaml` | 100K | 4-6h | Best quality, longer training |

**Key optimizations in fast configs:**
- Reduced timesteps: 200K → 50K (75% reduction)
- Reduced epochs: 8 → 4 (50% faster, also optimal)
- Smaller rollouts: 128 → 64 (faster collection)
- Larger batch size: 16 → 32 (faster updates on Colab T4)

### 6. Phase 3: Evaluation

**Evaluate trained policy:**
```powershell
py -3.11 scripts/evaluate_ppo.py `
    --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt `
    --probe_path checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ `
    --num_samples 50 `
    --device cuda
```

**Output location:**
- Results: `outputs/evaluation/eval_YYYYMMDD_HHMMSS/`
  - `evaluation_metrics.json` - Detailed metrics
  - `evaluation_summary.json` - Summary with targets
  - `sample_comparisons.png` - Visual comparison

## Full Pipeline (All Phases)

**Run complete pipeline from scratch:**
```powershell
# 1. Setup (one-time)
py -3.11 scripts/test_setup.py

# 2. Phase 1: Linear Probing (Nudity Focus)
py -3.11 scripts/collect_latents.py `
    --num_samples 50 `
    --num_steps 20 `
    --device cuda `
    --focus_nudity `
    --hard_only `
    --min_nudity_pct 50.0 `
    --min_inappropriate_pct 60.0 `
    --model_id CompVis/stable-diffusion-v1-4

# 2.1. Train probes (with improved heuristics)
py -3.11 scripts/train_probes.py --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS

# 2.2. (Optional) Measure empirical sensitivity for better accuracy
py -3.11 scripts/measure_layer_sensitivity.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --num_samples 20 `
    --device cuda

# 2.3. Re-train probes with empirical measurements
py -3.11 scripts/train_probes.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --use_empirical

# 2.5. Verify Probe Results (NEW - Recommended!)
# Generate images and visualize probe accuracy
py -3.11 scripts/generate_images_from_latents.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --timestep 8
# Open viewer.html in browser to see generated images
py -3.11 scripts/visualize_probe_results.py `
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS `
    --probe_dir ./checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ `
    --timestep 4

# 3. Phase 2: PPO Training
py -3.11 scripts/train_ppo.py --config configs/train_ppo_best.yaml

# 4. Phase 3: Evaluation
py -3.11 scripts/evaluate_ppo.py `
    --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt `
    --probe_path checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ `
    --num_samples 50
```

## Clearing Cache

**Clear HuggingFace cache:**
```powershell
Remove-Item -Recurse -Force data/cache/*
```

**Clear all generated data (latents, checkpoints, outputs):**
```powershell
Remove-Item -Recurse -Force data/latents/*
Remove-Item -Recurse -Force checkpoints/probes/*
Remove-Item -Recurse -Force outputs/ppo/*
Remove-Item -Recurse -Force outputs/evaluation/*
```

**Full clean (including cache):**
```powershell
Remove-Item -Recurse -Force data/cache/*
Remove-Item -Recurse -Force data/latents/*
Remove-Item -Recurse -Force checkpoints/probes/*
Remove-Item -Recurse -Force outputs/ppo/*
Remove-Item -Recurse -Force outputs/evaluation/*
```

## Troubleshooting

### CUDA Out of Memory
Use optimized config:
```powershell
py -3.11 scripts/train_ppo.py --config configs/rtx4050_optimized.yaml
```

### Module Not Found
Make sure you're in the project root:
```powershell
cd project-aether
py -3.11 scripts/test_setup.py
```

### Model Download Fails
Pre-download the model:
```powershell
py -3.11 -c "from diffusers import StableDiffusionPipeline; import torch; pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)"
```

### Images Not Generated or Look Wrong
- Make sure you're using the same model that collected latents
- Check the timestep (use final timestep for final images, e.g., timestep 8 for 8-step generation)
- If images weren't saved during collection, use `generate_images_from_latents.py` to decode them

### Probe Visualization Errors
- Ensure you've run `train_probes.py` first
- Check that probe directory contains `probe_t{timestep:02d}.pt` files
- Use the correct timestep (check `sensitivity_scores.json` for best timestep)

## Expected Results

### Phase 1 Targets
- Probe accuracy: >85% at optimal timesteps
- Layer sensitivity: Optimal intervention window identified
- **Verification:** Safe/unsafe images match their labels (use visualization scripts)

### Phase 2 Targets
- Training completes without crashes
- Policy loss decreases over time
- Final policy saved successfully

### Phase 3 Targets
- SSR (Safety Success Rate): >80% (unsafe→safe conversion)
- FPR (False Positive Rate): <5% (safe→flagged rate)
- LPIPS: <0.30 (perceptual distance)
- Transport Cost: Minimized

## Notes

- All commands use `py -3.11` for Python 3.11
- Replace `YYYYMMDD_HHMMSS` with actual timestamps from your runs
- GPU recommended for training (CPU is very slow)
- First run will download models (~4GB) - this takes 10-20 minutes
- **Model Selection:** `CompVis/stable-diffusion-v1-4` is recommended for research as it's less censored than SD 1.5
- **Content Focus:** Project now focuses on **nudity only** (not gore/violence) for clearer concept boundaries
- **Filtering:** Default settings use strict thresholds (≥50% nudity, ≥60% inappropriate, hard prompts only)
- **NEW:** Images are now saved automatically during latent collection (use `--no_save_images` to disable)
- **NEW:** Use visualization scripts to verify probe accuracy and see what images were actually generated
- See `PROBE_VISUALIZATION_GUIDE.md` for detailed visualization instructions

## Model Information

### Stable Diffusion 1.4 (CompVis) - Recommended

**Why SD 1.4 for this research:**
- Less censored than SD 1.5, allowing generation of nudity content needed for safety research
- Well-documented and stable
- Compatible with standard diffusers library
- Good balance between quality and research needs

**Model Details:**
- **HuggingFace ID:** `CompVis/stable-diffusion-v1-4`
- **License:** OpenRAIL
- **Size:** ~4GB
- **Inference Steps:** 20-50 recommended (slower than LCM but necessary for uncensored content)

**Alternative Models (Not Recommended):**
- `runwayml/stable-diffusion-v1-5`: More censored, may not generate needed content
- SDXL models: Different architecture, requires more VRAM
- Community uncensored models: Less stable, may have compatibility issues

**For Research Purposes:**
This project uses SD 1.4 for academic research on AI safety and alignment. The model is used to:
1. Study how diffusion models represent unsafe concepts in latent space
2. Develop RL-based steering methods to prevent unsafe generation
3. Evaluate safety mechanisms without retraining base models

All generated content is used solely for research and evaluation purposes.

