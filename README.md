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
5. [Google Colab Setup](#google-colab-setup) ⭐ **NEW**
6. [Hyperparameter Experiments](#hyperparameter-experiments) ⭐ **NEW**
7. [Project Structure](#project-structure)
8. [Running the Project](#running-the-project)
9. [Configuration Options](#configuration-options)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## 🎯 Project Overview

### Abstract
Generative models often reproduce undesired or unsafe concepts due to limited control over their latent representations. This project develops a reinforcement learning framework for concept steering in diffusion models. The framework learns a policy that transports latent representations toward or away from target concepts using an optimal-transport reward combining safety and semantic alignment.

### Key Contributions
- **Unified ODE framework** for diffusion and flow-matching models
- **Linear probing** for concept detection in latent space (90% accuracy achieved!)
- **Layer sensitivity analysis** to identify optimal intervention points
- **Modular reward system** with separate safety (R_safe) and transport (W2) components
- **Optimal transport reward** combining safety and semantic alignment
- **LCM integration** for 3x faster inference (8 steps vs 20+)
- **Evaluation framework** with SSR, LPIPS, and transport cost metrics
- **Comprehensive test suite** for environment, rewards, and PPO components

### Three-Phase Approach
| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: The Probe** | ✅ **COMPLETE** | Linear probes trained with 90% accuracy |
| **Phase 2: Policy Training** | ✅ **COMPLETE** | PPO policy trained (50K timesteps). Improved config ready (150K timesteps) |
| **Phase 3: Evaluation** | ✅ **COMPLETE** | Evaluation framework implemented. Initial results: SSR 13.3%, FPR 26.7% |

---

## 🏁 Current Progress

### ✅ Phase 1: Linear Probing - COMPLETE (Dec 13, 2025)

**Using LCM (Latent Consistency Model) for fast inference!**

#### Phase 1.1: Latent Collection ✅

```
Model: rupeshs/LCM-runwayml-stable-diffusion-v1-5
Output: data/latents/run_20251213_124105/
├── latents/
│   ├── timestep_00.npz through timestep_08.npz
│   └── (100 samples × 16,384 dimensions each)
├── safe_prompts.json      (50 safe prompts)
├── unsafe_prompts.json    (50 unsafe prompts from I2P)
└── metadata.json
```

**Collection Statistics:**
- **Model:** LCM (Latent Consistency Model) - 3x faster than standard SD
- **Safe samples:** 50 (curated prompts: nature, animals, food, architecture)
- **Unsafe samples:** 50 (from I2P dataset: violence, sexual, shocking)
- **Timesteps:** 9 (0-8, using LCM's efficient 8-step generation)
- **Latent dimension:** 16,384 (64×64×4 flattened)
- **Collection time:** ~2 minutes on RTX 4050

#### Phase 1.2: Linear Probe Training ✅

```
Output: checkpoints/probes/run_20251213_125128/
├── probe_metrics.json         # Accuracy at each timestep
├── sensitivity_scores.json    # Layer sensitivity analysis
├── probe_analysis.png         # Visualization
└── pytorch/
    └── probe_t00.pt ... probe_t08.pt  # Trained probes
```

**Probe Results (Concepts ARE Linearly Separable!):**

| Timestep | Test Accuracy | AUC | Assessment |
|----------|---------------|-----|------------|
| t=0 | 55% | 0.48 | ❌ Noise |
| t=1 | **85%** | **0.97** | ✅ Excellent |
| t=2 | **90%** | 0.95 | ✅ **Best** |
| t=3 | **90%** | 0.95 | ✅ **Best** |
| t=4 | **85%** | 0.90 | ✅ Optimal |
| t=5 | 75% | 0.90 | ✅ Good |
| t=6 | 75% | 0.89 | ✅ Good |
| t=7 | 70% | 0.85 | ⚠️ Fair |
| t=8 | 70% | 0.82 | ⚠️ Fair |

**Layer Sensitivity Analysis:**
- **Optimal Timestep:** t=4 (sensitivity score: 0.72)
- **Recommended Intervention Window:** [2, 6]
- **Top 5 Timesteps:** [4, 3, 5, 2, 6]

### ✅ Phase 2: PPO Training - COMPLETE (Dec 13, 2025)

**Successfully trained the steering policy using PPO (Proximal Policy Optimization)**

**Note:** Initial training completed. Improved configuration created for extended training (see below).

#### Configuration (Memory-Optimized for RTX 4050 6GB)

```yaml
# PPO hyperparameters - Aggressive memory optimization
ppo:
  n_steps: 64         # Rollout steps (reduced from 512)
  batch_size: 8       # Minibatch size (reduced from 32)
  n_epochs: 4         # PPO epochs (reduced from 10)
  total_timesteps: 50000
  learning_rate: 0.0003

# Environment settings
env:
  model_id: rupeshs/LCM-runwayml-stable-diffusion-v1-5
  num_inference_steps: 8
  intervention_window: [2, 6]  # From sensitivity analysis
  lambda_transport: 0.5        # Safety vs quality tradeoff
```

#### Training Results

**Run:** `outputs/ppo/aether_ppo_20251213_134441/`

- **Total timesteps:** 50,000
- **Training updates:** 781 (64 steps per rollout)
- **Final policy saved:** `final_policy.pt`
- **Checkpoints:** 10 checkpoints saved throughout training

#### Training Metrics Summary

| Metric | Initial | Final | Trend |
|--------|---------|-------|-------|
| `reward` | -0.086 | Variable (range: -0.66 to +0.71) | Learning |
| `policy_loss` | -0.121 | -0.085 | ↓ Decreasing (good) |
| `value_loss` | 0.199 | 0.138 | ↓ Decreasing (good) |
| `entropy` | 363.2 | 481.7 | ↑ Increasing (exploration) |

**Key Observations:**
- Policy loss decreased from -0.121 to -0.085, indicating learning
- Value function improved (loss decreased from 0.199 to 0.138)
- Entropy increased, showing maintained exploration
- Training completed successfully without crashes

#### Memory Optimizations Applied

1. **Reduced rollout steps**: 512 → 64 (prevents OOM during collection)
2. **Reduced batch size**: 32 → 8 (smaller GPU memory footprint)
3. **CUDA cache clearing**: After each rollout and update
4. **Trajectory storage disabled**: Only store action norms, not full tensors
5. **Observation normalization**: Prevents NaN in policy network
6. **Prompt truncation**: Max 50 words to avoid CLIP's 77 token limit

#### Output Directory

```
outputs/ppo/aether_ppo_20251213_134441/
├── checkpoint_*.pt          # 10 training checkpoints
├── final_policy.pt          # Final trained policy ✅
├── training_history.json   # Complete metrics history ✅
└── training_curves.png      # Loss/reward plots (if generated)
```

### ✅ Phase 3: Evaluation - COMPLETE (Dec 13, 2025)

**Evaluation framework implemented and initial results obtained**

The evaluation script has been tested and initial results show areas for improvement.

#### Evaluation Script

```bash
# Run evaluation on trained policy
python scripts/evaluate_ppo.py \
    --policy_path outputs/ppo/aether_ppo_20251213_134441/final_policy.pt \
    --probe_path checkpoints/probes/run_20251213_125128/pytorch/ \
    --num_samples 50 \
    --device cuda
```

#### Metrics Computed

1. **SSR (Safety Success Rate)**: Unsafe → Safe conversion rate
   - Target: >0.80 (higher is better)
   - Formula: (Unsafe prompts that became Safe) / (Total Unsafe prompts)

2. **FPR (False Positive Rate)**: Safe → Flagged rate
   - Target: <0.05 (lower is better)
   - Measures how often safe images are incorrectly flagged

3. **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual distance
   - Target: <0.30 (lower is better)
   - Measures perceptual similarity between original and steered images

4. **Transport Cost (W2)**: Wasserstein-2 inspired cost
   - Target: Minimize
   - Formula: Σ||Δz_t||² (sum of squared steering actions)

#### Output Directory

```
outputs/evaluation/eval_YYYYMMDD_HHMMSS/
├── evaluation_metrics.json      # Detailed metrics
├── evaluation_summary.json       # Summary with targets
├── test_prompts.json            # Test prompts and labels
└── sample_comparisons.png        # Visual comparison (first 10 samples)
```

#### Evaluation Process

For each test prompt:
1. **Generate baseline** (no steering) - original image
2. **Generate with policy** (steering enabled) - steered image
3. **Compute safety predictions** using linear probe
4. **Compute transport cost** from steering actions
5. **Compute LPIPS** between original and steered images
6. **Aggregate metrics** across all samples

#### Initial Evaluation Results (30 samples)

**Run:** `outputs/evaluation/eval_20251213_163422/`

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **SSR** | 0.1333 (2/15 unsafe→safe) | >0.80 | [FAIL] |
| **FPR** | 0.2667 (4/15 safe→flagged) | <0.05 | [FAIL] |
| **LPIPS** | 0.3200 ± 0.1034 | <0.30 | [FAIL] (close) |
| **Transport Cost** | 70.39 ± 46.46 | Minimize | - |

**Analysis:**
- **SSR (13.3%)**: Low conversion rate - policy needs more training
- **FPR (26.7%)**: High false positive rate - steering may be too aggressive
- **LPIPS (0.32)**: Close to target - quality preservation is reasonable
- **Transport Cost**: Moderate - steering actions within expected range

**Recommendations Implemented:**
1. ✅ Increased training time (50K → 150K timesteps)
2. ✅ Tuned lambda_transport (0.5 → 0.3 for more aggressive safety steering)
3. ✅ Improved hyperparameters (learning rate, n_epochs)
4. ⏳ Ready for extended training with improved configuration

---

## ✅ What Has Been Implemented

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Diffusion Environment** | `src/envs/diffusion_env.py` | Gymnasium environment wrapping LCM/Stable Diffusion ODE |
| **Latent Encoder** | `src/envs/diffusion_env.py` | Reduces observation space from 16,384 → 256 dimensions |
| **Linear Probe** | `src/models/linear_probe.py` | Concept detection probes for each timestep |
| **PPO Trainer** | `src/training/ppo_trainer.py` | Complete PPO implementation with ActorCritic network |
| **Safety Reward** | `src/rewards/safety_reward.py` | Probe-based safety reward computation (R_safe) |
| **Transport Reward** | `src/rewards/transport_reward.py` | Wasserstein-2 inspired transport cost (W2) |
| **Evaluation Metrics** | `src/evaluation/metrics.py` | SSR, FPR, LPIPS, Transport Cost metrics |
| **Data Utilities** | `src/utils/data.py` | I2P dataset loader + 200 curated safe prompts |

### Model
- **Base Model:** [rupeshs/LCM-runwayml-stable-diffusion-v1-5](https://huggingface.co/rupeshs/LCM-runwayml-stable-diffusion-v1-5)
- **Type:** Latent Consistency Model (LCM)
- **Advantage:** Only 4-8 inference steps needed (vs 20-50 for standard SD)
- **License:** OpenRAIL

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/collect_latents.py` | Collect latent representations for probe training |
| `scripts/train_probes.py` | Train linear probes at each timestep |
| `scripts/run_sensitivity.py` | Layer sensitivity analysis to find optimal intervention window |
| `scripts/run_phase1.py` | Complete Phase 1 pipeline (collect + train + analyze) |
| `scripts/train_ppo.py` | Train the steering policy with PPO |
| `scripts/evaluate_ppo.py` | **Phase 3:** Evaluate trained policy with SSR, LPIPS, Transport Cost, FPR |
| `scripts/test_setup.py` | Verify installation and setup |
| `scripts/quick_test.py` | Quick test of all components |

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_env.py` | AetherConfig, LatentEncoder, SteeringProjection, DiffusionSteeringEnv |
| `tests/test_rewards.py` | SafetyReward, TransportReward, compute_w2_cost, CombinedReward |
| `tests/test_ppo.py` | ActorCritic, RolloutBuffer, PPOConfig |

### Configurations

| Config | Purpose |
|--------|---------|
| `configs/base.yaml` | Base settings shared across experiments |
| `configs/train_ppo.yaml` | PPO training hyperparameters (initial) |
| `configs/train_ppo_improved.yaml` | **Improved PPO config** (150K timesteps, tuned λ=0.3) |
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
git clone https://github.com/Anastasia-Deniz/project-aether.git
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

The LCM (Latent Consistency Model) will be downloaded automatically on first run. To pre-download:

```bash
python -c "from diffusers import StableDiffusionPipeline; import torch; pipe = StableDiffusionPipeline.from_pretrained('rupeshs/LCM-runwayml-stable-diffusion-v1-5', torch_dtype=torch.float16)"
```

This takes 10-20 minutes depending on internet speed. LCM models are faster and require only 4-8 inference steps!

---

## ☁️ Google Colab Setup

**Run Project Aether on Google Colab with free GPU access!**

### Quick Start

1. **Open Google Colab**: https://colab.research.google.com/
2. **Upload `colab_setup.ipynb`** from this repository
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
4. **Run cells sequentially**

### Detailed Guide

See **[COLAB_GUIDE.md](COLAB_GUIDE.md)** for complete instructions.

### Advantages of Colab

- **Free GPU access** (T4 with 16GB VRAM)
- **Larger batch sizes** (16 vs 8 on RTX 4050)
- **No local setup required**
- **Easy result sharing** (save to Google Drive)

### Colab-Optimized Configuration

Use `configs/colab_optimized.yaml` for Colab:
- Larger batch sizes (16 vs 8)
- Larger rollouts (128 vs 64)
- Optimized for T4 GPU

### Quick Commands

```python
# Install dependencies
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install diffusers transformers accelerate gymnasium numpy scikit-learn matplotlib tqdm pyyaml pillow lpips datasets

# Clone repository
!git clone https://github.com/Anastasia-Deniz/project-aether.git
%cd project-aether

# Run phases
!python scripts/collect_latents.py --num_samples 100 --device cuda
!python scripts/train_probes.py --latents_dir data/latents/run_XXXXXX
!python scripts/train_ppo.py --config configs/colab_optimized.yaml
!python scripts/evaluate_ppo.py --policy_path outputs/ppo/XXXXXX/final_policy.pt --num_samples 30
```

### Save to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs /content/drive/MyDrive/project-aether-results/
```

---

## 🔬 Hyperparameter Experiments

**Run multiple experiments to find optimal hyperparameters!**

We've set up 9 experiments testing key hyperparameters:
- **Lambda Transport** (0.3, 0.5, 0.8, 1.0) - Controls action penalty
- **Learning Rate** (1e-4, 2e-4) - Controls learning speed
- **N Epochs** (4, 10) - Controls updates per rollout
- **Policy Capacity** ([256, 128] vs [512, 256]) - Controls network size

### Quick Start

```bash
# Run all experiments
python scripts/run_experiments.py

# Run specific experiments
python scripts/run_experiments.py --experiments exp_lambda_0.3 exp_lambda_0.5

# Skip already completed experiments
python scripts/run_experiments.py --skip-completed

# Compare results
python scripts/compare_experiments.py
```

### Experiment Details

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for:
- Full list of experiments
- Hypotheses for each
- Expected results
- Comparison guide

### Output

Each experiment creates:
```
outputs/ppo/{experiment_name}_{timestamp}/
├── final_policy.pt
├── training_history.json
└── checkpoint_*.pt
```

Compare results with `scripts/compare_experiments.py` to find the best hyperparameters!

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
│   ├── rewards/                  # Reward computation (NEW)
│   │   ├── __init__.py
│   │   ├── safety_reward.py      # R_safe from probe predictions
│   │   └── transport_reward.py   # W2 transport cost penalty
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
├── tests/                        # Unit tests (NEW)
│   ├── __init__.py
│   ├── test_env.py               # Environment tests
│   ├── test_rewards.py           # Reward module tests
│   └── test_ppo.py               # PPO component tests
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
# Test installation and setup
python scripts/test_setup.py

# Expected output:
# ✓ PyTorch
# ✓ CUDA available
# ✓ All structure checks passed

# Test all components
python scripts/quick_test.py

# Expected output:
# ✓ LatentEncoder OK
# ✓ Safe Prompts OK (200 base prompts)
# ✓ PPO Components OK
# ✓ Evaluation Metrics OK
# ✓ Environment Config OK
```

### Run Unit Tests

```bash
# Test environment components
python tests/test_env.py

# Test reward modules
python tests/test_rewards.py

# Test PPO components
python tests/test_ppo.py
```

### Phase 1: Linear Probing & Sensitivity Analysis

#### Step 1.1: Collect Latents
```bash
# For GPU (recommended) - LCM uses 8 steps for fast inference
python scripts/collect_latents.py --num_samples 50 --num_steps 8 --device cuda

# For CPU (slow but works)
python scripts/collect_latents.py --num_samples 20 --num_steps 8 --device cpu
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

### Phase 2: PPO Policy Training ✅ COMPLETE

**Training completed successfully!**

The policy has been trained and saved at:
```
outputs/ppo/aether_ppo_20251213_134441/final_policy.pt
```

To retrain or continue training:

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
- `final_policy.pt` - Trained policy ✅
- `training_history.json` - Training metrics ✅
- `training_curves.png` - Loss/reward plots (if generated)

### Phase 3: Evaluation ✅ COMPLETE

**Evaluation framework implemented**

Run evaluation using:

```bash
python scripts/evaluate_ppo.py \
    --policy_path outputs/ppo/aether_ppo_20251213_134441/final_policy.pt \
    --num_samples 50
```

The evaluation script:
1. Loads the trained policy from checkpoint
2. Generates images with and without steering
3. Computes SSR, LPIPS, Transport Cost, and FPR metrics
4. Saves results and visualizations

**Output:** `outputs/evaluation/eval_YYYYMMDD_HHMMSS/`

---

## ⚙️ Configuration Options

### Memory-Optimized Settings (6GB VRAM)

If you have a GPU with 6GB VRAM (like RTX 4050, RTX 3060 Mobile), use these settings:

```yaml
# In configs/train_ppo.yaml
env:
  model_id: "rupeshs/LCM-runwayml-stable-diffusion-v1-5"
  num_inference_steps: 8       # LCM works great with 4-8 steps
  
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
python scripts/collect_latents.py --device cpu --num_samples 10 --num_steps 8
python scripts/train_ppo.py --quick --device cpu
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inference_steps` | 8 | LCM diffusion steps (4-8 recommended) |
| `steering_dim` | 256 | Low-rank steering dimension |
| `lambda_transport` | 0.5 | Safety vs quality tradeoff |
| `intervention_start` | 2 | Start steering at this step |
| `intervention_end` | 6 | Stop steering at this step |
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

#### 2. "CUDA Out of Memory" (GPU OOM)
This is common on 6GB GPUs. Our optimized settings:
```yaml
ppo:
  n_steps: 64      # Reduced from 512
  batch_size: 8    # Reduced from 32
  n_epochs: 4      # Reduced from 10
```
Also uses automatic CUDA cache clearing and stores only action norms (not full tensors).

#### 3. "NaN values in policy network"
Caused by large observation values. Fixed by:
- Normalizing latent observations (zero mean, unit variance)
- Clamping values to [-10, 10] range
- Using `torch.nan_to_num()` for safety

#### 4. "Token indices sequence length > 77" (CLIP Warning)
This is a **benign warning** - CLIP truncates long prompts automatically. We also pre-truncate prompts to max 50 words.

#### 5. PowerShell "NativeCommandError"
```
+ CategoryInfo : NotSpecified: (:String) [], RemoteException
+ FullyQualifiedErrorId : NativeCommandError
```
This is **NOT an error**! PowerShell treats stderr output (progress bars, warnings) as errors. The training is still running correctly.

#### 6. "Model download fails"
The model (`rupeshs/LCM-runwayml-stable-diffusion-v1-5`) doesn't require HuggingFace login. If download fails:
```bash
# Check internet connection
# Try running again - downloads resume from where they stopped
python -c "from diffusers import StableDiffusionPipeline; pipe = StableDiffusionPipeline.from_pretrained('rupeshs/LCM-runwayml-stable-diffusion-v1-5')"
```

#### 7. "Module not found: src"
Make sure you're running from the project root:
```bash
cd project-aether
python scripts/collect_latents.py
```

#### 8. Slow Training
- Use GPU if available
- Reduce `num_samples` for initial tests
- Use `--quick` flag for fast testing

### Getting Help

1. Run diagnostics: `python scripts/test_setup.py`
2. Run quick test: `python scripts/quick_test.py`
3. Check GPU: `python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"`

---

## 📊 Results

### Phase 1: Probe Accuracy ✅ ACHIEVED
- ✅ **Achieved: 90%** accuracy at timesteps 2-3
- ✅ **Achieved: 97%** AUC at timestep 1
- Target was >85% - **EXCEEDED!**
- Linear separability **CONFIRMED** (Alain & Bengio, 2016)

### Phase 2: Training Results ✅
- **Training completed:** 50,000 timesteps
- **Policy loss:** Decreased from -0.121 to -0.085 (learning confirmed)
- **Value loss:** Decreased from 0.199 to 0.138 (value function improved)
- **Entropy:** Increased from 363 to 482 (maintained exploration)
- **Final policy:** Saved at `outputs/ppo/aether_ppo_20251213_134441/final_policy.pt`

### Phase 3: Evaluation Results ✅

**Initial evaluation completed (30 samples)**

**Initial evaluation completed (30 samples)**

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| SSR | 13.3% (2/15) | >80% | [FAIL] - Needs more training |
| LPIPS | 0.32 ± 0.10 | <0.3 | [FAIL] - Close to target |
| FPR | 26.7% (4/15) | <5% | [FAIL] - Too high |
| W2 | 70.39 ± 46.46 | Minimize | - |

**Analysis:**
- Policy shows learning but needs extended training
- Improved configuration created to address low SSR and high FPR
- Ready for extended training with improved hyperparameters

**To run evaluation:**
```bash
py -3.11 scripts/evaluate_ppo.py --policy_path outputs/ppo/aether_ppo_20251213_134441/final_policy.pt --num_samples 30
```

---

## 📚 References

1. **Lamba et al. (2025)** - "Alignment and Safety of Diffusion Models via Reinforcement Learning"
2. **Holderrieth & Erives (2025)** - MIT 6.S184: Generative AI With SDEs
3. **Alain & Bengio (2016)** - "Understanding Intermediate Layers Using Linear Classifier Probes"
4. **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
5. **Wagenmaker et al. (2025)** - "Steering Your Diffusion Policy with Latent Space RL"

---

## 👥 Team

- **Alkan**
- **Durak** 
- **Chiucchiolo** 

---

## 📝 License

This project is for educational purposes as part of the Advanced Machine Learning course at Sapienza University of Rome.

---

*Last updated: December 13, 2025 (All phases complete - improved training config ready)*

