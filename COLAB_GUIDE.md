# Project Aether - Google Colab Setup Guide

Complete guide to run Project Aether on Google Colab with GPU support.

## Quick Start

1. **Open Google Colab**: https://colab.research.google.com/
2. **Upload `colab_setup.ipynb`** or copy the cells below
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
4. **Run cells sequentially**

---

## Step-by-Step Setup

### Step 1: Install Dependencies

```python
# Install PyTorch with CUDA
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
!pip install diffusers transformers accelerate safetensors
!pip install gymnasium numpy scikit-learn matplotlib tqdm
!pip install pyyaml pillow lpips
!pip install datasets  # For I2P dataset
```

### Step 2: Clone or Upload Repository

**Option A: Clone from GitHub**
```python
!git clone https://github.com/Anastasia-Deniz/project-aether.git
%cd project-aether
```

**Option B: Upload Files**
1. Upload project files to Colab
2. Unzip if needed: `!unzip project-aether.zip`
3. `%cd project-aether`

### Step 3: Verify Setup

```python
import torch
import sys
from pathlib import Path

# Check GPU
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Setup paths
sys.path.insert(0, str(Path.cwd()))
!mkdir -p data/latents checkpoints/probes outputs/ppo outputs/evaluation
```

---

## Running Each Phase

### Phase 1: Collect Latents

```python
# Colab T4 has 16GB VRAM - can use more samples
# Using --hard_only and higher threshold for better probe accuracy (target: >85%)
!python scripts/collect_latents.py \
    --num_samples 150 \
    --num_steps 8 \
    --device cuda \
    --min_inappropriate_pct 70.0 \
    --hard_only
```

**Note**: If probe accuracy is <80%, see `COLAB_PROBE_IMPROVEMENTS.md` for troubleshooting.

**Output:** `data/latents/run_YYYYMMDD_HHMMSS/`

### Phase 1: Train Probes

```python
# Find latest latents directory
import os
from pathlib import Path

latents_dirs = sorted(Path('data/latents').glob('run_*'), key=os.path.getmtime)
latest_latents = latents_dirs[-1]
print(f"Using: {latest_latents}")

!python scripts/train_probes.py --latents_dir {latest_latents}
```

**Output:** `checkpoints/probes/run_YYYYMMDD_HHMMSS/`

### Phase 2: Train PPO Policy

```python
# Use Colab-optimized config (larger batches)
# Config uses probe_path: "auto" to automatically find latest probe
!python scripts/train_ppo.py --config configs/colab_optimized.yaml
```

**Or manually specify probe path:**
```python
import os
from pathlib import Path

# Find latest probe
probe_dirs = sorted(Path('checkpoints/probes').glob('run_*'), key=os.path.getmtime)
latest_probe = probe_dirs[-1] / 'pytorch'

!python scripts/train_ppo.py \
    --config configs/colab_optimized.yaml \
    --probe_path {latest_probe}
```

**Or use V2 config:**
```python
!python scripts/train_ppo.py --config configs/train_ppo_v2.yaml
```

**Output:** `outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/`

### Phase 3: Evaluate

```python
# Find latest policy
import os
from pathlib import Path

ppo_dirs = sorted(Path('outputs/ppo').glob('aether_ppo_*'), key=os.path.getmtime)
latest_policy = ppo_dirs[-1] / 'final_policy.pt'

# Find latest probe
probe_dirs = sorted(Path('checkpoints/probes').glob('run_*'), key=os.path.getmtime)
latest_probe = probe_dirs[-1] / 'pytorch'

if latest_policy.exists() and latest_probe.exists():
    !python scripts/evaluate_ppo.py \
        --policy_path {latest_policy} \
        --probe_path {latest_probe} \
        --num_samples 30 \
        --device cuda
else:
    print(f"Policy exists: {latest_policy.exists()}")
    print(f"Probe exists: {latest_probe.exists()}")
```

---

## Colab-Specific Features

### Save to Google Drive

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Copy results
!cp -r outputs /content/drive/MyDrive/project-aether-results/
!cp -r checkpoints /content/drive/MyDrive/project-aether-results/
!cp -r data /content/drive/MyDrive/project-aether-results/
```

### Download Results

```python
from google.colab import files
import zipfile
import os

# Create zip
with zipfile.ZipFile('aether_results.zip', 'w') as zipf:
    for root, dirs, files in os.walk('outputs'):
        for file in files:
            zipf.write(os.path.join(root, file))
    for root, dirs, files in os.walk('checkpoints'):
        for file in files:
            zipf.write(os.path.join(root, file))

# Download
files.download('aether_results.zip')
```

### Monitor Training Progress

```python
# Check latest training directory
import json
from pathlib import Path

ppo_dirs = sorted(Path('outputs/ppo').glob('aether_ppo_*'), key=os.path.getmtime)
if ppo_dirs:
    latest = ppo_dirs[-1]
    history_file = latest / 'training_history.json'
    
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        
        print(f"Training: {latest.name}")
        print(f"Total updates: {len(history['rewards'])}")
        print(f"Latest reward: {history['rewards'][-1]:.4f}")
        print(f"Avg last 10: {sum(history['rewards'][-10:])/10:.4f}")
```

---

## Colab Optimizations

### Memory Management

```python
# Clear cache periodically
import torch
import gc

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Use after each phase
clear_memory()
```

### Faster Model Loading

```python
# Pre-download model (run once)
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    'rupeshs/LCM-runwayml-stable-diffusion-v1-5',
    torch_dtype=torch.float16
)
```

### Checkpoint Management

```python
# List all checkpoints
from pathlib import Path

ppo_dir = Path('outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS')
checkpoints = sorted(ppo_dir.glob('checkpoint_*.pt'))
print(f"Found {len(checkpoints)} checkpoints")
for ckpt in checkpoints[-5:]:  # Last 5
    print(f"  {ckpt.name}")
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solution:** Use smaller batch sizes
```python
# Edit config or use command line
!python scripts/train_ppo.py \
    --config configs/train_ppo_v2.yaml \
    --batch_size 8 \
    --n_steps 64
```

### Model Download Fails

**Solution:** Use HuggingFace cache
```python
import os
os.environ['HF_HOME'] = '/content/.cache/huggingface'
```

### Session Timeout

**Solution:** Save checkpoints to Drive
```python
# Run periodically during training
from google.colab import drive
drive.mount('/content/drive')
!cp outputs/ppo/*/checkpoint_*.pt /content/drive/MyDrive/backups/
```

---

## Expected Performance

**Colab T4 GPU (16GB VRAM):**
- **Phase 1 (Collect Latents)**: ~5-10 minutes (100 samples)
- **Phase 1 (Train Probes)**: ~2-3 minutes
- **Phase 2 (PPO Training)**: ~3-4 hours (200K timesteps)
- **Phase 3 (Evaluation)**: ~10-15 minutes (30 samples)

**Advantages over RTX 4050:**
- Larger batch sizes (16 vs 8)
- Larger rollouts (128 vs 64)
- No memory constraints
- Free GPU access

---

## Quick Reference

```python
# Complete workflow
!python scripts/collect_latents.py --num_samples 100 --device cuda
!python scripts/train_probes.py --latents_dir data/latents/run_XXXXXX
!python scripts/train_ppo.py --config configs/colab_optimized.yaml
!python scripts/evaluate_ppo.py --policy_path outputs/ppo/XXXXXX/final_policy.pt --num_samples 30
```

---

## Notes

- **Colab sessions timeout after ~12 hours** - save checkpoints to Drive
- **Free tier has usage limits** - consider Colab Pro for longer training
- **Files are deleted when session ends** - always save to Drive
- **GPU type varies** - T4 is common, sometimes V100 or A100

