# Model Switch: LCM SD 1.5 → CompVis SD 1.4

## Why the Switch?

**Problem:** Stable Diffusion 1.5 (and LCM based on it) may have content filtering that prevents generating unsafe content needed for training.

**Solution:** Switched to **CompVis/stable-diffusion-v1-4**, the original uncensored version.

## Changes Made

### Model
- **Old:** `rupeshs/LCM-runwayml-stable-diffusion-v1-5` (LCM, 4-8 steps)
- **New:** `CompVis/stable-diffusion-v1-4` (SD 1.4, 20-50 steps)

### Inference Steps
- **Old:** 8 steps (LCM optimized)
- **New:** 20 steps (SD 1.4 standard)

### Trade-offs
- ✅ **Better:** Less censored, can generate unsafe content
- ⚠️ **Slower:** 20 steps vs 8 steps (~2.5x slower)
- ✅ **Better for training:** Actually generates unsafe images from unsafe prompts

## Updated Files

All config files have been updated:
- `configs/base.yaml`
- `configs/train_ppo*.yaml`
- `configs/experiments/*.yaml`
- `configs/colab_optimized.yaml`
- `src/envs/diffusion_env.py` (default)
- `scripts/collect_latents.py` (default)

## Testing

To verify the model generates unsafe content:

```bash
py -3.11 scripts/test_unsafe_generation.py --model_id "CompVis/stable-diffusion-v1-4"
```

## Note

SD 1.4 requires more steps (20+) than LCM (4-8), so training will be slower but necessary for generating the unsafe content needed for the safety steering task.

