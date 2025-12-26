# A100 GPU Configuration Guide

## Overview

Three optimized configurations for Google Colab A100 GPU (40GB VRAM):

1. **Fast** (`colab_a100_fast.yaml`) - 1-2 hours, quick experiments
2. **Optimized** (`colab_a100_optimized.yaml`) - 2-3 hours, recommended
3. **Best** (`colab_a100_best.yaml`) - 3-4 hours, maximum quality

## A100 vs T4 Comparison

| Feature | T4 (16GB) | A100 (40GB) | Improvement |
|---------|-----------|-------------|-------------|
| VRAM | 16GB | 40GB | 2.5x |
| Batch Size | 32 | 128 | 4x |
| Rollout Size | 64-128 | 256-512 | 2-4x |
| Policy Network | [512, 256] | [1024, 512, 256] | 2x capacity |
| Training Speed | 1x | 3-5x | 3-5x faster |

## Configuration Details

### Fast Config (1-2 hours)
- **Timesteps**: 50,000
- **Batch Size**: 128
- **Rollout Size**: 256
- **Epochs**: 4
- **Policy**: [512, 256]
- **Use Case**: Quick experiments, testing, debugging

### Optimized Config (2-3 hours) ⭐ **RECOMMENDED**
- **Timesteps**: 100,000
- **Batch Size**: 128
- **Rollout Size**: 256
- **Epochs**: 6
- **Policy**: [1024, 512, 256]
- **Use Case**: Standard training, good balance of speed and quality

### Best Config (3-4 hours)
- **Timesteps**: 200,000
- **Batch Size**: 128
- **Rollout Size**: 512
- **Epochs**: 8
- **Policy**: [1024, 512, 256]
- **Early Stopping**: Enabled
- **Use Case**: Final training, maximum quality, publications

## Auto-Detection

The Colab notebook automatically detects A100 and selects the appropriate config:

```python
# Auto-detection logic:
if "A100" in gpu_name or vram_gb >= 35:
    # Use A100 configs
else:
    # Use T4 configs
```

## Manual Selection

To manually select a config in the notebook, edit Step 7:

```python
# Change these flags:
use_fast = True   # For fast config
use_best = True   # For best config
# Leave both False for optimized (default)
```

## Performance Expectations

### A100 Fast
- **Time**: 1-2 hours
- **Final Performance**: Good (suitable for testing)
- **Memory Usage**: ~20-25GB VRAM

### A100 Optimized ⭐
- **Time**: 2-3 hours
- **Final Performance**: Very Good (recommended)
- **Memory Usage**: ~25-30GB VRAM

### A100 Best
- **Time**: 3-4 hours
- **Final Performance**: Excellent (publication quality)
- **Memory Usage**: ~30-35GB VRAM

## Key Optimizations

1. **Larger Batch Sizes**: 4x larger batches (32 → 128) = faster updates
2. **Larger Rollouts**: 2-4x larger rollouts (64-128 → 256-512) = better sample efficiency
3. **Larger Policy Network**: 2x capacity ([512,256] → [1024,512,256]) = better representation
4. **More Epochs**: Can afford more epochs (4 → 6-8) = better convergence
5. **More Timesteps**: Can afford more timesteps (50K → 100K-200K) = better final performance

## Memory Usage

A100 configs are designed to use:
- **Fast**: ~20-25GB VRAM (leaves headroom)
- **Optimized**: ~25-30GB VRAM (efficient use)
- **Best**: ~30-35GB VRAM (near maximum)

This leaves 5-10GB headroom for system overhead.

## Recommendations

1. **Start with Optimized**: Best balance of speed and quality
2. **Use Fast for Testing**: Quick iterations during development
3. **Use Best for Final Training**: Maximum quality for results/papers
4. **Monitor Memory**: Check `nvidia-smi` if you encounter OOM errors
5. **Adjust Batch Size**: If OOM, reduce `batch_size` in config (128 → 96 → 64)

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size`: 128 → 96 → 64
- Reduce `n_steps`: 256 → 128
- Use Fast config instead

### Training Too Slow
- Use Fast config
- Reduce `total_timesteps`: 100K → 50K
- Reduce `n_epochs`: 6 → 4

### Want Better Quality
- Use Best config
- Increase `total_timesteps`: 100K → 200K
- Increase `n_epochs`: 6 → 8

## Comparison with T4

| Metric | T4 Fast | A100 Fast | A100 Optimized | A100 Best |
|--------|---------|-----------|----------------|-----------|
| Time | 2-3h | 1-2h | 2-3h | 3-4h |
| Batch Size | 32 | 128 | 128 | 128 |
| Rollout | 64 | 256 | 256 | 512 |
| Timesteps | 50K | 50K | 100K | 200K |
| Policy | [512,256] | [512,256] | [1024,512,256] | [1024,512,256] |
| Quality | Good | Good | Very Good | Excellent |

## Notes

- All configs use `float16` for efficiency
- All configs use `probe_path: "auto"` to auto-detect latest probe
- All configs use intervention window [5, 15] for 20-step generation
- Early stopping is enabled only in Best config

