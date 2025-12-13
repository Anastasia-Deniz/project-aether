# Improving Probe Accuracy in Colab

Your probe training achieved **70% accuracy**, which is below the ideal 85%+ target. Here's how to improve it.

## Why Lower Accuracy?

The 70% accuracy suggests:
- **Less distinct prompts**: The safe/unsafe prompts may be too similar
- **Lower inappropriate threshold**: Default 30% might include borderline cases
- **Random seed differences**: Different prompts selected than your local run

## Solutions

### Option 1: Use Harder Prompts (Recommended)

Collect latents with **harder, more clearly inappropriate** prompts:

```python
# In Colab, modify the collect_latents command:
!python scripts/collect_latents.py \
    --num_samples 100 \
    --num_steps 8 \
    --device cuda \
    --min_inappropriate_pct 70.0 \
    --hard_only
```

**What this does:**
- `--min_inappropriate_pct 70.0`: Only prompts with >70% inappropriate rating
- `--hard_only`: Only "hard" prompts (most clearly inappropriate)

**Expected improvement:** 75-85% accuracy

### Option 2: Collect More Samples

More data can help the probe learn better:

```python
!python scripts/collect_latents.py \
    --num_samples 200 \
    --num_steps 8 \
    --device cuda \
    --min_inappropriate_pct 50.0
```

**Expected improvement:** 72-78% accuracy

### Option 3: Use Specific Categories

Focus on the most distinct categories:

```python
!python scripts/collect_latents.py \
    --num_samples 100 \
    --num_steps 8 \
    --device cuda \
    --categories violence sexual \
    --min_inappropriate_pct 60.0
```

### Option 4: Proceed with 70% (Quick Option)

**70% accuracy is still usable!** The probe will work, just with:
- More false positives/negatives
- Less reliable safety detection
- Still sufficient for PPO training

You can proceed to Phase 2 training, but expect:
- Lower SSR (Safety Success Rate)
- Higher FPR (False Positive Rate)
- Policy may need more training to compensate

## Recommended Colab Workflow

```python
# Step 1: Collect with harder prompts
!python scripts/collect_latents.py \
    --num_samples 150 \
    --num_steps 8 \
    --device cuda \
    --min_inappropriate_pct 70.0 \
    --hard_only

# Step 2: Train probes
import os
from pathlib import Path
latents_dirs = sorted(Path('data/latents').glob('run_*'), key=os.path.getmtime)
latest_latents = latents_dirs[-1]
!python scripts/train_probes.py --latents_dir {latest_latents}

# Step 3: Check accuracy
# If >80%, proceed to PPO training
# If 70-80%, proceed but expect lower performance
# If <70%, try Option 1 again with different parameters
```

## Understanding the Results

Your current results:
```
Best accuracy: 0.700 (70%) at t=1
Optimal intervention window: [2, 6]
```

**What this means:**
- ✅ **Probe is learning**: 70% > 50% (random)
- ⚠️ **Moderate separability**: Safe/unsafe latents are somewhat distinct
- ✅ **Intervention window identified**: Steps 2-6 are best for steering

**Can you proceed?** Yes, but:
- Use the probes from `checkpoints/probes/run_20251213_230403/pytorch/`
- Expect lower SSR in evaluation (maybe 10-20% instead of 30%+)
- May need more PPO training to compensate

## Quick Fix Script

Add this cell to your Colab notebook:

```python
# Improved latent collection for better probe accuracy
!python scripts/collect_latents.py \
    --num_samples 150 \
    --num_steps 8 \
    --device cuda \
    --min_inappropriate_pct 70.0 \
    --hard_only \
    --categories violence sexual shocking

# Train probes
import os
from pathlib import Path
latents_dirs = sorted(Path('data/latents').glob('run_*'), key=os.path.getmtime)
if latents_dirs:
    latest_latents = latents_dirs[-1]
    print(f"Training probes on: {latest_latents}")
    !python scripts/train_probes.py --latents_dir {latest_latents}
else:
    print("No latents found!")
```

## Expected Results After Improvement

With `--min_inappropriate_pct 70.0 --hard_only`:
- **Target accuracy**: 80-90%
- **If still low**: Check prompt quality or try different categories
- **If >85%**: Excellent! Proceed to PPO training

---

**Note**: The current 70% probes will work for training, but improving them will lead to better final policy performance.

