# Project Aether: Step-by-Step Execution Guide

## Executive Summary

Your project is **well-designed** with a solid theoretical foundation. The main risks are computational cost and the linear separability assumption. Follow this guide to mitigate risks and execute efficiently.

---

## Phase 0: Environment Setup (Days 1-2)

### Local Machine Configuration

```bash
# 1. Create conda environment
conda create -n aether python=3.10 -y
conda activate aether

# 2. Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install requirements
pip install -r requirements.txt

# 4. Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Cursor Setup

1. **Open the project folder** in Cursor
2. **Enable the AI features** for code completion
3. **Create a `.cursor` folder** with these rules:

```json
// .cursor/settings.json
{
  "ai.codeCompletion.enabled": true,
  "ai.chat.contextFiles": ["src/**/*.py", "configs/*.yaml"]
}
```

4. **Use Cursor Chat** for:
   - "Explain this function"
   - "Optimize this for GPU"
   - "Add type hints to this file"

### Download Base Model

```python
# run_setup.py - Downloads the model once
from diffusers import StableDiffusionPipeline
import torch

# Start with SD 2.1 base (smaller, faster than SDXL)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16,
)
pipe.save_pretrained("./models/sd-2.1-base")
print("Model downloaded and saved!")
```

---

## Phase 1: The Probe (Days 3-7)

**Goal**: Validate that unsafe concepts are linearly separable in the latent space and find the optimal intervention timestep.

### Step 1.1: Prepare Your Prompts

```python
# data/prompts.py
SAFE_PROMPTS = [
    "a beautiful sunset over the ocean",
    "a cat sleeping on a couch",
    "children playing in a park",
    "a professional headshot of a businesswoman",
    # ... 500+ prompts from MS-COCO captions
]

UNSAFE_PROMPTS = [
    # Get these from I2P benchmark
    # Categories: violence, gore, nudity (as per your doc)
    # WARNING: Handle these carefully, don't include explicit text
    # Use prompt IDs/hashes instead of raw text
]
```

### Step 1.2: Collect Latents

```bash
# Run latent collection (this takes time)
python scripts/collect_latents.py \
    --model_path ./models/sd-2.1-base \
    --safe_prompts data/safe_prompts.txt \
    --unsafe_prompts data/unsafe_prompts.txt \
    --num_steps 20 \
    --output_dir ./data/latents/
```

**Expected output**: `./data/latents/timestep_{0..20}.npz` files

### Step 1.3: Train Linear Probes

```bash
python scripts/train_probes.py \
    --latents_dir ./data/latents/ \
    --output_dir ./checkpoints/probes/
```

**What to look for**:
- If accuracy > 85% at mid-timesteps → Good, proceed
- If accuracy < 70% everywhere → Problem: concepts may not be linearly separable
  - **Mitigation**: Try nonlinear probes (small MLP) or focus on coarser categories

### Step 1.4: Layer Sensitivity Analysis

```bash
python scripts/run_sensitivity.py \
    --probes_dir ./checkpoints/probes/ \
    --output sensitivity_results.json
```

**Expected Result**: Identify optimal timestep range (likely t ≈ 8-12 out of 20)

### Phase 1 Deliverable Checklist

- [ ] Probe accuracy plot (Figure 1 from your doc)
- [ ] Optimal intervention timestep identified
- [ ] Written analysis: "Concepts X and Y are separable at t=..."

---

## Phase 2: Policy Training (Days 8-21)

### Step 2.1: Configure Training

```yaml
# configs/train_ppo.yaml
env:
  model_path: ./models/sd-2.1-base
  num_inference_steps: 20
  steering_dim: 256  # Low-rank

policy:
  hidden_dims: [512, 256]
  activation: relu

ppo:
  learning_rate: 3e-4
  n_steps: 2048  # Steps per update
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  clip_range: 0.2
  total_timesteps: 500_000  # Start small

reward:
  lambda_transport: 0.5  # Adjust based on Phase 1
  safety_classifier: ./checkpoints/safety_classifier.pth
  
intervention:
  start_timestep: 8   # From sensitivity analysis
  end_timestep: 14    # From sensitivity analysis
```

### Step 2.2: Start with Single Concept

Don't try multi-concept steering immediately. Start with one:

```bash
# Train on nudity detection first (usually clearest signal)
python scripts/train_ppo.py \
    --config configs/train_nudity.yaml \
    --concept nudity \
    --wandb_project aether-phase2
```

### Step 2.3: Monitor Training

Watch these metrics in W&B:
- `rollout/ep_rew_mean`: Should increase
- `train/policy_loss`: Should decrease
- `custom/safety_rate`: Should increase
- `custom/transport_cost`: Should stay bounded

**Warning signs**:
- Reward explodes → Reward hacking, increase λ
- Reward stuck → Learning rate too low or probe is bad
- High variance → Reduce action magnitude

### When to Move to Cloud Compute

Move to cloud when:
1. Local training takes >1 hour per 10K timesteps
2. You need to run hyperparameter sweeps
3. You want to scale to larger models

---

## Compute Options Comparison

| Platform | GPU | Cost | Best For |
|----------|-----|------|----------|
| **Kaggle** | P100 or T4 | Free (30h/week) | Phase 1, small experiments |
| **Colab Pro** | T4/V100 | $10/month | Prototyping, medium runs |
| **Lambda Labs** | A10/A100 | $0.60-$1.10/hr | Serious training |
| **RunPod** | Various | $0.20-$0.80/hr | Flexible, good value |
| **Vast.ai** | Various | $0.10-$0.50/hr | Cheapest, less reliable |

### My Recommendation

1. **Days 1-7 (Phase 1)**: Local machine (if you have GPU) or Kaggle free tier
2. **Days 8-14**: Colab Pro or Kaggle for initial PPO experiments
3. **Days 15-21**: Lambda Labs or RunPod for full training runs

### Kaggle Setup

```python
# In a Kaggle notebook
!pip install diffusers transformers accelerate stable-baselines3

# Mount your data from Kaggle Datasets
import kagglehub
# Upload your latents as a dataset
```

### Lambda Labs Setup

```bash
# SSH into instance
ssh ubuntu@<instance-ip>

# Clone your repo
git clone https://github.com/yourusername/project-aether.git
cd project-aether

# Setup environment
conda create -n aether python=3.10 -y
conda activate aether
pip install -r requirements.txt

# Run training with tmux (persists after disconnect)
tmux new -s training
python scripts/train_ppo.py --config configs/train_ppo.yaml
# Ctrl+B, D to detach
```

---

## Phase 3: Evaluation (Days 22-28)

### Step 3.1: Benchmark Suite

```bash
# Run full evaluation
python scripts/evaluate.py \
    --model_path ./models/sd-2.1-base \
    --policy_path ./checkpoints/policies/best_policy.pth \
    --unsafe_prompts data/i2p_test.txt \
    --safe_prompts data/mscoco_test.txt \
    --output_dir ./outputs/evaluation/
```

### Step 3.2: Metrics to Report

| Metric | Formula | Target |
|--------|---------|--------|
| **SSR** | (Unsafe→Safe) / Total Unsafe | > 80% |
| **LPIPS** | Perceptual distance | < 0.3 |
| **W2** | Σ‖Δz_t‖² | Minimize |
| **FPR** | (Safe→Flagged) / Total Safe | < 5% |

### Step 3.3: Baselines to Compare

1. **Unsteered**: Raw model output (floor)
2. **Negative prompts**: Add "not violent, not nude" to prompt
3. **Safe Latent Diffusion**: If you can implement
4. **Your method**: Aether

### Step 3.4: Ablations

- λ = 0 (pure safety, no transport cost)
- λ = 1.0 (high quality preservation)
- Steering at all timesteps vs. optimal window only
- Low-rank dim = 64 vs 256 vs 512

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Setup + Phase 1 | Environment, latent collection, probe training |
| 2 | Phase 1 → 2 | Sensitivity analysis, initial PPO experiments |
| 3 | Phase 2 | Full PPO training, hyperparameter tuning |
| 4 | Phase 3 | Evaluation, ablations, write-up |

---

## Common Pitfalls and Solutions

### Pitfall 1: "My probe accuracy is low everywhere"
**Cause**: Concepts aren't linearly separable at this level
**Solution**: 
- Try probing on UNet hidden states instead of just latents
- Use a small MLP probe (2 layers) instead of linear
- Focus on more distinct concepts (nudity vs. landscape, not "subtle violence")

### Pitfall 2: "Training is too slow"
**Cause**: Full diffusion rollouts are expensive
**Solution**:
- Reduce inference steps to 10
- Use float16 everywhere
- Only intervene at optimal timesteps (not all 20)
- Move to cloud

### Pitfall 3: "Reward hacking - images become garbage but 'safe'"
**Cause**: λ is too low
**Solution**:
- Increase λ from 0.5 to 1.0 or higher
- Add LPIPS directly to reward
- Constrain action magnitude: clip ‖Δz‖ < 0.1

### Pitfall 4: "Policy doesn't learn"
**Cause**: Multiple possibilities
**Solution**:
- Check reward scale (should be roughly -1 to +1)
- Verify environment is working (render intermediate images)
- Try simpler policy architecture
- Increase n_steps for more stable updates

---

## Quick Reference: Key Commands

```bash
# Phase 1
python scripts/collect_latents.py --num_samples 500
python scripts/train_probes.py
python scripts/run_sensitivity.py

# Phase 2
python scripts/train_ppo.py --config configs/train_ppo.yaml

# Phase 3
python scripts/evaluate.py --output_dir ./outputs/final/

# Utilities
python -m wandb login  # Setup experiment tracking
tensorboard --logdir ./outputs/logs  # Local monitoring
```

---

## Final Notes

Your project is ambitious but achievable. The key insight—layer sensitivity analysis—is genuinely novel and could make a solid contribution. Focus on:

1. **Validating the SSB hypothesis empirically** (this will make or break the project)
2. **Starting simple** (one concept, one model, then scale)
3. **Comparing against baselines** (professors love ablations)

Good luck! Feel free to ask follow-up questions as you progress.
