# Project Aether: Latent Space Transport Control

> **Advanced Machine Learning Course Project**  
> Sapienza University of Rome - Fall 2025  
> Team: Alkan, Durak, Chiucchiolo

A Reinforcement Learning framework for safe concept steering in diffusion models. This project learns a policy that transports latent representations toward or away from target concepts using an optimal-transport reward combining safety and semantic alignment.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What Has Been Implemented](#what-has-been-implemented)
3. [System Requirements](#system-requirements)
4. [Installation Guide](#installation-guide)
5. [Project Structure](#project-structure)
6. [Running the Project](#running-the-project)
7. [Configuration Options](#configuration-options)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## 🎯 Project Overview

### Abstract
Generative models often reproduce undesired or unsafe concepts due to limited control over their latent representations. This project develops a reinforcement learning framework for concept steering in diffusion models. The framework learns a policy that transports latent representations toward or away from target concepts using an optimal-transport reward combining safety and semantic alignment.

### Key Contributions
- **Unified ODE framework** for diffusion and flow-matching models
- **Linear probing** for concept detection in latent space
- **Layer sensitivity analysis** to identify optimal intervention points
- **Optimal transport reward** combining safety and semantic alignment
- **Evaluation framework** with SSR, LPIPS, and transport cost metrics

### Three-Phase Approach
| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: The Probe** | ✅ Latent Collection Done | Train linear classifiers to detect unsafe concepts |
| **Phase 2: Policy Training** | ⏳ Pending | Train PPO policy to steer latents toward safe generations |
| **Phase 3: Evaluation** | ⏳ Pending | Benchmark against baselines using SSR, LPIPS, and transport cost |

---

## 🏁 Current Progress

### ✅ Phase 1.1: Latent Collection - COMPLETED (Dec 8, 2025)

Successfully collected latent representations for training probes:

```
Output: data/latents/run_20251208_224020/
├── latents/
│   ├── timestep_00.npz through timestep_16.npz
│   └── (100 samples × 16,384 dimensions each)
├── safe_prompts.json      (50 safe prompts)
├── unsafe_prompts.json    (50 unsafe prompts from I2P)
└── metadata.json
```

**Statistics:**
- **Safe samples:** 50 (curated prompts: nature, animals, food, architecture)
- **Unsafe samples:** 50 (from I2P dataset: violence, hate, sexual content)
- **Timesteps:** 17 (0-16, covering full diffusion trajectory)
- **Latent dimension:** 16,384 (64×64×4 flattened)
- **Collection time:** ~4 minutes on RTX 4050

### ⏳ Next Steps
1. **Train Linear Probes** - Run `scripts/train_probes.py`
2. **Sensitivity Analysis** - Run `scripts/run_sensitivity.py`
3. **PPO Training** - Run `scripts/train_ppo.py`

---

## ✅ What Has Been Implemented

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Diffusion Environment** | `src/envs/diffusion_env.py` | Gymnasium environment wrapping Stable Diffusion ODE |
| **Latent Encoder** | `src/envs/diffusion_env.py` | Reduces observation space from 16,384 → 256 dimensions |
| **Linear Probe** | `src/models/linear_probe.py` | Concept detection probes for each timestep |
| **PPO Trainer** | `src/training/ppo_trainer.py` | Complete PPO implementation with ActorCritic network |
| **Evaluation Metrics** | `src/evaluation/metrics.py` | SSR, FPR, LPIPS, Transport Cost metrics |
| **Data Utilities** | `src/utils/data.py` | I2P dataset loader + 200 curated safe prompts |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/collect_latents.py` | Collect latent representations for probe training |
| `scripts/train_probes.py` | Train linear probes at each timestep |
| `scripts/run_sensitivity.py` | Layer sensitivity analysis to find optimal intervention window |
| `scripts/train_ppo.py` | Train the steering policy with PPO |
| `scripts/test_setup.py` | Verify installation and setup |
| `scripts/quick_test.py` | Quick test of all components |

### Configurations

| Config | Purpose |
|--------|---------|
| `configs/base.yaml` | Base settings shared across experiments |
| `configs/train_ppo.yaml` | PPO training hyperparameters |
| `configs/collect_latents.yaml` | Latent collection settings |
| `configs/rtx4050_optimized.yaml` | Settings optimized for 6GB VRAM GPUs |

---

## 💻 System Requirements

### Minimum Requirements (CPU Mode)
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or 3.11 (recommended: 3.11)
- **RAM**: 16 GB
- **Storage**: 10 GB free space (for model downloads)

### Recommended Requirements (GPU Mode)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060, RTX 4050, etc.)
- **CUDA**: 11.8 or 12.x
- **RAM**: 16 GB
- **Storage**: 15 GB free space

### Tested Configurations
| Hardware | Status | Notes |
|----------|--------|-------|
| RTX 4050 Laptop (6GB) | ✅ Tested | Use optimized configs |
| RTX 3060 (12GB) | ✅ Should work | Can use default configs |
| RTX 4090 (24GB) | ✅ Should work | Can use larger batch sizes |
| CPU Only | ✅ Tested | Very slow, for development only |

---

## 🔧 Installation Guide

### Step 1: Clone the Repository

```bash
git clone <repository-url>](https://github.com/Anastasia-Deniz/project-aether.git)
cd project-aether
```

### Step 2: Create Python Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n aether python=3.11 -y
conda activate aether
```

**Option B: Using venv**
```bash
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate
```

### Step 3: Install PyTorch

**For NVIDIA GPU (CUDA 12.4):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**For NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only:**
```bash
pip install torch torchvision
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Run setup test
python scripts/test_setup.py

# Expected output:
# ✓ PyTorch
# ✓ CUDA available (or ⚠ CPU mode)
# ✓ All structure checks passed
```

### Step 6: Download the Model (First Time Only)

The Stable Diffusion model (~4GB) will be downloaded automatically on first run. To pre-download:

```bash
python -c "from diffusers import StableDiffusionPipeline; import torch; pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)"
```

This takes 15-30 minutes depending on internet speed.

---

## 📁 Project Structure

```
project-aether/
│
├── configs/                      # Configuration files
│   ├── base.yaml                 # Base settings
│   ├── train_ppo.yaml            # PPO training config
│   ├── collect_latents.yaml      # Latent collection config
│   └── rtx4050_optimized.yaml    # Optimized for 6GB GPUs
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── envs/                     # RL Environment
│   │   ├── __init__.py
│   │   └── diffusion_env.py      # Main Gymnasium environment
│   │
│   ├── models/                   # Model components
│   │   ├── __init__.py
│   │   └── linear_probe.py       # Linear probes for concept detection
│   │
│   ├── training/                 # Training modules
│   │   ├── __init__.py
│   │   └── ppo_trainer.py        # PPO implementation
│   │
│   ├── evaluation/               # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py            # SSR, LPIPS, Transport Cost
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── data.py               # Dataset loading
│
├── scripts/                      # Executable scripts
│   ├── collect_latents.py        # Phase 1: Collect latents
│   ├── train_probes.py           # Phase 1: Train probes
│   ├── run_sensitivity.py        # Phase 1: Sensitivity analysis
│   ├── train_ppo.py              # Phase 2: Train policy
│   ├── test_setup.py             # Verify installation
│   └── quick_test.py             # Quick component test
│
├── data/                         # Data directory
│   ├── cache/                    # HuggingFace cache
│   └── latents/                  # Collected latents
│
├── checkpoints/                  # Saved models
│   ├── probes/                   # Trained probes
│   └── policies/                 # Trained policies
│
├── outputs/                      # Generated outputs
│   ├── images/                   # Generated images
│   └── logs/                     # Training logs
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Running the Project

### Quick Start (Test Everything Works)

```bash
# Test all components
python scripts/quick_test.py

# Expected output:
# ✓ LatentEncoder OK
# ✓ Safe Prompts OK (200 base prompts)
# ✓ PPO Components OK
# ✓ Evaluation Metrics OK
# ✓ Environment Config OK
```

### Phase 1: Linear Probing & Sensitivity Analysis

#### Step 1.1: Collect Latents
```bash
# For GPU (recommended)
python scripts/collect_latents.py --num_samples 50 --num_steps 15 --device cuda

# For CPU (very slow, ~2 hours)
python scripts/collect_latents.py --num_samples 20 --num_steps 10 --device cpu
```

**Output:** `data/latents/run_YYYYMMDD_HHMMSS/`

#### Step 1.2: Train Linear Probes
```bash
python scripts/train_probes.py --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS
```

**Output:** `checkpoints/probes/run_YYYYMMDD_HHMMSS/`
- `probe_metrics.json` - Accuracy at each timestep
- `pytorch/probe_t*.pt` - Trained probe models
- `probe_analysis.png` - Visualization

#### Step 1.3: Layer Sensitivity Analysis
```bash
python scripts/run_sensitivity.py --probes_dir ./checkpoints/probes/run_YYYYMMDD_HHMMSS
```

**Output:** `outputs/sensitivity/run_YYYYMMDD_HHMMSS/`
- `sensitivity_results.json` - Optimal intervention window
- `sensitivity_analysis.png` - Layer sensitivity plot (Figure 1)

### Phase 2: PPO Policy Training

```bash
# For GPU with 6GB VRAM
python scripts/train_ppo.py --config configs/train_ppo.yaml

# Quick test (CPU or GPU)
python scripts/train_ppo.py --quick

# With custom settings
python scripts/train_ppo.py \
    --total_timesteps 100000 \
    --lambda_transport 0.5 \
    --device cuda
```

**Output:** `outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/`
- `final_policy.pt` - Trained policy
- `training_history.json` - Training metrics
- `training_curves.png` - Loss/reward plots

### Phase 3: Evaluation

*Coming soon - evaluate steering performance*

---

## ⚙️ Configuration Options

### Memory-Optimized Settings (6GB VRAM)

If you have a GPU with 6GB VRAM (like RTX 4050, RTX 3060 Mobile), use these settings:

```yaml
# In configs/train_ppo.yaml
env:
  num_inference_steps: 15      # Reduced from 20
  
ppo:
  n_steps: 512                 # Reduced from 2048
  batch_size: 32               # Reduced from 64
  
policy:
  hidden_dims: [256, 128]      # Reduced from [512, 256]
```

Or use the pre-configured optimized config:
```bash
python scripts/train_ppo.py --config configs/rtx4050_optimized.yaml
```

### CPU Mode Settings

For development without GPU:
```bash
python scripts/collect_latents.py --device cpu --num_samples 10 --num_steps 5
python scripts/train_ppo.py --quick --device cpu
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inference_steps` | 15 | Diffusion steps (lower = faster, less quality) |
| `steering_dim` | 256 | Low-rank steering dimension |
| `lambda_transport` | 0.5 | Safety vs quality tradeoff |
| `intervention_start` | 6 | Start steering at this step |
| `intervention_end` | 11 | Stop steering at this step |
| `n_steps` | 512 | PPO rollout length |
| `batch_size` | 32 | Minibatch size |
| `total_timesteps` | 100000 | Training duration |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "CUDA not available"
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### 2. "Out of Memory" (GPU)
- Reduce `num_inference_steps` to 10
- Reduce `batch_size` to 16
- Reduce `n_steps` to 256
- Use `configs/rtx4050_optimized.yaml`

#### 3. "Model download fails"
The model (`runwayml/stable-diffusion-v1-5`) doesn't require HuggingFace login. If download fails:
```bash
# Check internet connection
# Try running again - downloads resume from where they stopped
python -c "from diffusers import StableDiffusionPipeline; pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

#### 4. "Module not found: src"
Make sure you're running from the project root:
```bash
cd project-aether
python scripts/collect_latents.py
```

#### 5. Slow Training
- Use GPU if available
- Reduce `num_samples` for initial tests
- Use `--quick` flag for fast testing

### Getting Help

1. Run diagnostics: `python scripts/test_setup.py`
2. Run quick test: `python scripts/quick_test.py`
3. Check GPU: `python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"`

---

## 📊 Expected Results

### Phase 1: Probe Accuracy
- Good: >85% accuracy at mid-timesteps
- Moderate: 70-85% accuracy
- Poor: <70% (concepts may not be linearly separable)

### Phase 2: Training Metrics
- `rollout/ep_rew_mean`: Should increase
- `train/policy_loss`: Should decrease
- `custom/safety_rate`: Should increase
- `custom/transport_cost`: Should stay bounded

### Phase 3: Evaluation Targets
| Metric | Target | Description |
|--------|--------|-------------|
| SSR | >80% | Safety Success Rate |
| LPIPS | <0.3 | Perceptual similarity |
| FPR | <5% | False Positive Rate |
| W2 | Minimize | Transport cost |

---

## 📚 References

1. **Lamba et al. (2025)** - "Alignment and Safety of Diffusion Models via Reinforcement Learning"
2. **Holderrieth & Erives (2025)** - MIT 6.S184: Generative AI With SDEs
3. **Alain & Bengio (2016)** - "Understanding Intermediate Layers Using Linear Classifier Probes"
4. **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
5. **Wagenmaker et al. (2025)** - "Steering Your Diffusion Policy with Latent Space RL"

---

## 👥 Team

- **Alkan** - [Role]
- **Durak** - [Role]
- **Chiucchiolo** - [Role]

---

## 📝 License

This project is for educational purposes as part of the Advanced Machine Learning course at Sapienza University of Rome.

---

*Last updated: December 8, 2025*

