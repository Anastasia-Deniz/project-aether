# Quick Fix for Colab Probe Path Error

## Problem
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/probes/run_20251213_125128/pytorch'
```

The config file has a hardcoded probe path that doesn't exist in Colab.

## Solution 1: Use Auto-Detection (Easiest)

The config files now support `probe_path: "auto"` which automatically finds the latest probe.

**Just run:**
```python
!python scripts/train_ppo.py --config configs/colab_optimized.yaml
```

The script will automatically find your latest probe at `checkpoints/probes/run_20251213_230403/pytorch/`

## Solution 2: Manually Set Probe Path

```python
import os
from pathlib import Path

# Find your latest probe (from your training output)
probe_dirs = sorted(Path('checkpoints/probes').glob('run_*'), key=os.path.getmtime)
latest_probe = probe_dirs[-1] / 'pytorch'

print(f"Using probe: {latest_probe}")

# Train with explicit probe path
!python scripts/train_ppo.py \
    --config configs/colab_optimized.yaml \
    --probe_path {latest_probe}
```

## Solution 3: Edit Config File

Edit `configs/colab_optimized.yaml` and change:
```yaml
reward:
  probe_path: "auto"  # Already set to auto-detect
```

Or set to your specific path:
```yaml
reward:
  probe_path: "./checkpoints/probes/run_20251213_230403/pytorch/"
```

## Verify Probe Path

Before training, check which probe you have:
```python
from pathlib import Path
import os

probe_dirs = sorted(Path('checkpoints/probes').glob('run_*'), key=os.path.getmtime)
for probe_dir in probe_dirs:
    pytorch_dir = probe_dir / 'pytorch'
    if pytorch_dir.exists():
        probe_files = list(pytorch_dir.glob('probe_t*.pt'))
        print(f"{probe_dir.name}: {len(probe_files)} probe files")
        if probe_files:
            print(f"  Example: {probe_files[0].name}")
```

---

**Recommended**: Use Solution 1 (auto-detection) - it's the easiest and works automatically!

