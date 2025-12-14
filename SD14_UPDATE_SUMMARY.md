# SD 1.4 Update Summary

## Changes Made for Stable Diffusion 1.4

### Model Switch
- **From:** `rupeshs/LCM-runwayml-stable-diffusion-v1-5` (LCM, 8 steps)
- **To:** `CompVis/stable-diffusion-v1-4` (SD 1.4, 20 steps)
- **Reason:** SD 1.4 is less censored and can actually generate unsafe content needed for training

### Intervention Window Update
- **From:** [2, 6] for 8 steps (~25% to 75%)
- **To:** [5, 15] for 20 steps (~25% to 75%)
- **Scaling:** Proportionally scaled from 8-step window

### Updated Files

**Configs (all updated):**
- All 9 experiment configs in `configs/experiments/`
- `configs/base.yaml`
- `configs/train_ppo*.yaml`
- `configs/colab_optimized.yaml`
- `configs/rtx4050_optimized.yaml`

**Code:**
- `src/envs/diffusion_env.py` (default model and intervention window)
- `scripts/train_ppo.py` (defaults)
- `scripts/collect_latents.py` (default model)

**Documentation:**
- `README.md` (model description)
- `COLAB_GUIDE.md` (updated steps and model info)
- `colab_setup.ipynb` (updated to 20 steps)

### Performance Impact

**Slower but necessary:**
- 20 steps vs 8 steps = ~2.5x slower per image
- But essential for generating unsafe content
- Training will take longer but will have proper unsafe data

### Next Steps

1. **Re-collect latents** with SD 1.4 to ensure unsafe content is generated
2. **Re-train probes** if needed (may get better accuracy with actual unsafe content)
3. **Run experiments** - they're now configured for SD 1.4

All experiments are ready to run with SD 1.4!

