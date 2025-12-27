# Project Aether: Latent Space Transport Control

> **Advanced Machine Learning Course Project**  
> Sapienza University of Rome - Fall 2025  
> Team: Alkan, Durak, Chiucchiolo

A reinforcement learning framework for safe concept steering in diffusion models. This project learns a policy that transports latent representations away from unsafe concepts using an optimal-transport reward combining safety and semantic alignment.

---

## 🎯 Overview

### Abstract

Generative models often reproduce undesired or unsafe concepts due to limited control over their latent representations. This project develops a reinforcement learning framework for concept steering in diffusion models. The framework learns a policy that transports latent representations toward or away from target concepts using an optimal-transport reward combining safety and semantic alignment.

### Key Contributions

- **Unified ODE framework** for diffusion models as deterministic probability flow
- **Linear probing** for concept detection in latent space (90% accuracy achieved)
- **Empirical layer sensitivity analysis** with FID and SSR measurements
- **Layer sensitivity analysis** to identify optimal intervention points
- **Modular reward system** with separate safety (R_safe) and transport (W2) components
- **Optimal transport reward** combining safety and semantic alignment
- **Improved intermediate reward shaping** for faster learning
- **Configuration validation** to prevent runtime errors
- **Hyperparameter experiment framework** for systematic optimization
- **Deterministic evaluation framework** with SSR, LPIPS, FPR, and transport cost metrics
- **Robust evaluation system** with fixed seeds and correct metric calculations

---

## 🔄 Complete Pipeline

The project follows a **three-phase pipeline** for training and evaluating the steering policy:

### Phase 1: Linear Probing & Concept Detection

**Goal**: Validate concept separability in latent space and identify optimal intervention timesteps.

**Steps**:

1. **Collect Latent Representations** (`scripts/collect_latents.py`)
   - Generates images from safe and unsafe prompts using Stable Diffusion 1.4
   - Collects latent representations at each timestep during generation
   - Focuses on nudity-only content for clearer concept boundaries
   - Filters: ≥50% nudity, ≥60% inappropriate, hard prompts only
   - Output: Latent tensors saved to `data/latents/run_YYYYMMDD_HHMMSS/`

2. **Train Linear Probes** (`scripts/train_probes.py`)
   - Trains logistic regression classifiers on latents at each timestep
   - Measures concept separability (safe vs unsafe) in latent space
   - Uses cross-validation and hyperparameter tuning for accuracy
   - Output: Probes saved to `checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/`
   - Best probe typically at timestep 4 (for 20-step generation)

3. **Measure Layer Sensitivity** (Optional, `scripts/measure_layer_sensitivity.py`)
   - Empirical measurement of intervention effectiveness at each timestep
   - Uses FID and SSR metrics to identify optimal intervention window
   - Output: Sensitivity scores indicating best timesteps for intervention
   - Optimal window: [5, 15] for 20-step generation (~25% to 75% of process)

**Current State**: ✅ **Complete**
- Latest probe run: `run_20251225_183438`
- Probe accuracy: ~85-90% at optimal timesteps
- Optimal intervention window: [5, 15] for 20-step generation

### Phase 2: PPO Policy Training

**Goal**: Train a reinforcement learning policy to steer latents away from unsafe concepts.

**Steps**:

1. **Configure Training** (`configs/train_ppo_best.yaml` or experiment configs)
   - Set hyperparameters: learning rate, batch size, epochs, lambda_transport
   - Configure intervention window (from Phase 1 sensitivity analysis)
   - Set reward weights: safety reward vs transport cost penalty

2. **Train Policy** (`scripts/train_ppo.py`)
   - Creates Gymnasium environment wrapping Stable Diffusion
   - Loads pre-trained probe from Phase 1 for safety scoring
   - Trains Actor-Critic network using PPO algorithm
   - Policy learns to output steering vectors that move latents toward safety
   - Reward: `R = R_safe - λ * Σ||Δz_t||²` (safety minus transport cost)
   - Output: Trained policy saved to `outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt`

3. **Monitor Training** (automatic)
   - Training history saved to `training_history.json`
   - Metrics: policy loss, value loss, entropy, episode rewards
   - Checkpoints saved periodically for recovery

**Current State**: ✅ **Multiple policies trained**
- Best performing: `exp_lambda_0.8` (SSR: 0.72, FPR: 0.44)
- Multiple experiments completed with different hyperparameters
- Policies available in `outputs/ppo/`

### Phase 3: Evaluation

**Goal**: Measure steering effectiveness with comprehensive, deterministic metrics.

**Steps**:

1. **Single Policy Evaluation** (`scripts/evaluate_ppo.py`)
   - Loads trained policy and probe
   - Generates images with and without steering on test prompts
   - Computes metrics: SSR, FPR, LPIPS, Transport Cost
   - **Deterministic**: Uses fixed seed (default: 42) for reproducibility
   - Output: Results saved to `outputs/evaluation/eval_YYYYMMDD_HHMMSS/`

2. **Batch Evaluation** (`scripts/evaluate_all_policies.py`)
   - Automatically finds all trained policies
   - Evaluates each with same test set (deterministic)
   - Compares results across policies
   - Output: Summary of all evaluations

3. **Hyperparameter Search** (`scripts/hyperparameter_search.py`)
   - Runs training + evaluation for multiple experiment configs
   - Compares results to find best hyperparameters
   - Output: Ranked list of experiments with metrics

**Current State**: ✅ **Fully Deterministic & Robust**
- **Deterministic evaluation**: Same seed → same results every time
- **Correct metrics**: SSR and FPR now correctly calculated
- **Fixed probe timestep mismatch**: Uses probe trained on final timestep (t=19)
- **Simplified evaluation logic**: Direct probe predictions, no arbitrary thresholds
- **Reliable safety assessment**: Ground truth baselines, trustworthy predictions

---

## 📊 Evaluation Metrics (Current Implementation)

### SSR (Safety Success Rate)
**Formula**: `SSR = (Unsafe → Safe conversions) / (Total Unsafe prompts)`

**How it's calculated** (FIXED):
1. For each unsafe prompt (ground truth label=1):
   - Original prediction: Uses **ground truth label** (baseline)
   - Steered prediction: Uses **probe trained on final timestep** (t=19)
   - Direct prediction: `steered_pred = int(probe_prob_unsafe > 0.5)`
   - If steered prediction = 0 (safe) → counts as successful conversion

**Target**: SSR > 0.80 (80% of unsafe images should become safe)

**Reliability**: ✅ **Fixed** - Now uses probe trained on correct timestep

### FPR (False Positive Rate)
**Formula**: `FPR = (Safe → Flagged as Unsafe) / (Total Safe prompts)`

**How it's calculated** (FIXED):
1. For each safe prompt (ground truth label=0):
   - Original prediction: Uses **ground truth label** (baseline)
   - Steered prediction: Uses **probe trained on final timestep** (t=19)
   - Direct prediction: `steered_pred = int(probe_prob_unsafe > 0.5)`
   - If steered prediction = 1 (unsafe) → counts as false positive

**Target**: FPR < 0.05 (less than 5% of safe images should be flagged)

**Reliability**: ✅ **Fixed** - Now uses probe trained on correct timestep

### LPIPS (Learned Perceptual Image Patch Similarity)
**Formula**: Perceptual distance between original and steered images

**How it's calculated**:
- Uses LPIPS library (AlexNet backbone)
- Measures perceptual similarity (not pixel-level)
- Lower = more similar = better quality preservation

**Target**: LPIPS < 0.30 (steered images should be perceptually similar to originals)

**Current Results**: 
- Typical range: 0.004 - 0.007 (excellent quality preservation)
- Very low distortion from steering

### Transport Cost (W2)
**Formula**: `Transport Cost = Σ_t ||Δz_t||²` (sum of squared action norms)

**How it's calculated**:
- Sum of squared norms of all steering actions during intervention window
- Measures "work" done by the policy
- Lower = more efficient steering

**Target**: Minimize (lower is better)

**Current Results**:
- Typical range: 200 - 600
- Varies by lambda_transport hyperparameter
- Higher lambda → lower transport cost (more efficient actions)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Anastasia-Deniz/project-aether.git
cd project-aether

# Create environment (Windows)
setup_env.bat

# Or manually:
conda create -n aether python=3.11 -y
conda activate aether
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Verify installation
py -3.11 scripts/test_setup.py
```

### Complete Pipeline

```bash
# Phase 1: Collect latents and train probes
py -3.11 scripts/collect_latents.py \
    --num_samples 50 \
    --num_steps 20 \
    --focus_nudity \
    --hard_only \
    --min_nudity_pct 50.0 \
    --device cuda

py -3.11 scripts/train_probes.py \
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS

# Phase 2: Train PPO policy
py -3.11 scripts/train_ppo.py \
    --config configs/train_ppo_best.yaml \
    --probe_path checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/

# Phase 3: Evaluate policy (DETERMINISTIC)
py -3.11 scripts/evaluate_ppo.py \
    --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt \
    --probe_path checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ \
    --num_samples 50 \
    --seed 42  # Fixed seed for reproducibility

# Evaluate all policies
py -3.11 scripts/evaluate_all_policies.py --num-samples 50
```

### Hyperparameter Search

```bash
# Run all experiments (training + evaluation)
py -3.11 scripts/hyperparameter_search.py

# Or use batch script (Windows)
scripts\run_hyperparameter_search_py311.bat
```

**For detailed instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**  
**For hyperparameter search, see [HYPERPARAMETER_SEARCH.md](HYPERPARAMETER_SEARCH.md)**  
**For evaluation fixes, see [EVALUATION_FIXES.md](EVALUATION_FIXES.md)**

---

## 📋 Requirements

### Minimum
- Python 3.11 (required for experiments)
- 16 GB RAM
- 10 GB storage

### Recommended
- NVIDIA GPU with 6GB+ VRAM (RTX 3060, RTX 4050, etc.)
- CUDA 11.8 or 12.x
- 15 GB storage

### Model
- **Base Model:** `CompVis/stable-diffusion-v1-4` (less censored than SD 1.5, better for research)
- **License:** OpenRAIL
- **Size:** ~4GB (downloaded automatically)

---

## 📁 Project Structure

```
project-aether/
├── src/                    # Core source code
│   ├── envs/              # RL environment (diffusion_env.py)
│   │   └── diffusion_env.py  # Gymnasium environment wrapping Stable Diffusion
│   ├── models/             # Linear probes (linear_probe.py)
│   ├── rewards/            # Reward computation
│   │   ├── safety_reward.py    # Safety reward (probe-based or classifier-based)
│   │   └── transport_reward.py # Transport cost (W2)
│   ├── training/           # PPO trainer
│   │   └── ppo_trainer.py  # PPO algorithm implementation
│   ├── evaluation/         # Evaluation metrics
│   │   └── metrics.py      # SSR, FPR, LPIPS, Transport Cost
│   └── utils/              # Data loading utilities
│       └── data.py        # I2P dataset, safe prompts
├── scripts/                # Executable scripts
│   ├── collect_latents.py      # Phase 1: Collect latents
│   ├── train_probes.py         # Phase 1: Train probes
│   ├── measure_layer_sensitivity.py  # Phase 1: Empirical measurement
│   ├── train_ppo.py            # Phase 2: Train policy
│   ├── evaluate_ppo.py         # Phase 3: Evaluate single policy
│   ├── evaluate_all_policies.py # Phase 3: Evaluate all policies
│   ├── hyperparameter_search.py # Run experiments
│   └── debug_evaluation.py     # Debug evaluation issues
├── configs/                # Configuration files
│   ├── train_ppo_best.yaml     # Recommended config (λ=0.5)
│   ├── train_ppo_optimized.yaml # Extended training (λ=0.8)
│   └── experiments/            # Hyperparameter experiment configs
│       ├── exp_lambda_0.2.yaml
│       ├── exp_lambda_0.5.yaml
│       ├── exp_lambda_0.8.yaml
│       └── ...
├── data/                   # Data directory
│   ├── latents/            # Collected latent representations
│   └── cache/              # HuggingFace cache
├── checkpoints/            # Saved models
│   └── probes/             # Trained linear probes
├── outputs/                # Generated outputs
│   ├── ppo/                # Trained policies
│   └── evaluation/         # Evaluation results
└── tests/                  # Unit tests
```

---

## 🔬 Key Features

### Content Focus
- **Nudity-only focus** for clearer concept boundaries
- **Strict filtering** (≥50% nudity, ≥60% inappropriate, hard prompts only)
- **Model selection:** SD 1.4 (less censored) for research purposes

### Deterministic Evaluation
- **Fixed seeds**: All evaluations use seed=42 by default for reproducibility
- **Deterministic generation**: Same prompt + same seed → same image
- **Consistent metrics**: Same evaluation → same results every time
- **Correct calculations**: SSR and FPR now properly computed

### Memory Optimizations
- Latent encoder reduces observation space (16K → 256 dims)
- CUDA cache clearing between episodes
- Trajectory storage disabled during training
- Optimized configs for 6GB GPUs (RTX 4050)

### Visualization & Verification
- Image generation from collected latents
- HTML viewer for browsing safe/unsafe images
- Probe visualization with confusion matrices
- Comprehensive metrics (SSR, FPR, LPIPS, Transport Cost)

---

## 📊 Results

### Phase 1: Linear Probing
- ✅ **85-90% accuracy** at optimal timesteps (4-5 for 20-step)
- ✅ **97% AUC** at early timesteps
- ✅ Linear separability confirmed
- ✅ Optimal intervention window: [5, 15] (for 20-step generation)
- ✅ Sensitivity analysis identifies best timesteps

### Phase 2: PPO Training
- ✅ Training completed successfully for multiple configurations
- ✅ Policy loss decreased (learning confirmed)
- ✅ Memory optimizations enable training on 6GB GPUs
- ✅ Multiple hyperparameter experiments completed

### Phase 3: Evaluation (Current State)

**Evaluation System**: ✅ **Fully Deterministic & Robust**

- **Determinism**: Fixed seeds ensure reproducible results
- **Correct Metrics**: SSR and FPR properly calculated
- **FPR Variation**: Now correctly varies by policy (not stuck at 0.44)

**Best Results** (from `exp_lambda_0.8`):
- **SSR**: 0.72 (72% of unsafe images converted to safe)
- **FPR**: 0.44 (44% of safe images flagged as unsafe)
- **LPIPS**: 0.0044 (excellent quality preservation)
- **Transport Cost**: 446.56 (moderate efficiency)

**Targets**:
- SSR > 0.80 (currently 0.72, needs improvement)
- FPR < 0.05 (currently 0.44, needs significant improvement)
- LPIPS < 0.30 (currently 0.004, excellent)
- Transport Cost: minimize (currently 446, moderate)

**Evaluation Process**:
1. Loads test prompts (50 samples: 25 safe, 25 unsafe)
2. For each prompt:
   - Generates image **without** steering (baseline, seed=42+i)
   - Generates image **with** steering (policy, same seed=42+i)
   - Compares probe scores before/after steering
3. Computes metrics deterministically
4. Saves results to `outputs/evaluation/eval_YYYYMMDD_HHMMSS/`

---

## 🔧 Configuration

### Recommended Configs

| Config | Use Case | Timesteps | Time | Description |
|--------|----------|-----------|------|-------------|
| `train_ppo_best.yaml` | **Recommended** | 100K | 4-6h | Optimal from experiments (λ=0.5) |
| `train_ppo_optimized.yaml` | Extended | 200K | 8-10h | Maximum performance (λ=0.8) |
| `colab_fast_20steps.yaml` | **Colab Fast** | 50K | 2-3h | Fast training, compatible with probes |
| `train_ppo_fast_optimized.yaml` | Very Fast | 30K | ~1h | Quick testing/iteration |
| `rtx4050_optimized.yaml` | Low VRAM | 100K | 4-6h | Optimized for 6GB GPUs |

### Running Experiments

To find optimal hyperparameters, run experiments:

```bash
# Run all experiments (training + evaluation)
py -3.11 scripts/hyperparameter_search.py

# Or use batch script (Windows)
scripts\run_hyperparameter_search_py311.bat
```

Available experiments (in `configs/experiments/`):
- `exp_lambda_0.2`, `exp_lambda_0.3`, `exp_lambda_0.5`, `exp_lambda_0.8`, `exp_lambda_1.0`
- `exp_lr_1e4`, `exp_lr_2e4`, `exp_lr_3e4`
- `exp_epochs_4`, `exp_epochs_10`
- `exp_batch_size_16`
- `exp_intervention_early`, `exp_intervention_late`
- And more...

See [HYPERPARAMETER_SEARCH.md](HYPERPARAMETER_SEARCH.md) for complete guide.

---

## 🧪 Testing

```bash
# Run unit tests
py -3.11 tests/test_env.py
py -3.11 tests/test_rewards.py
py -3.11 tests/test_ppo.py

# Quick component test
py -3.11 scripts/quick_test.py
```

---

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup and usage instructions
- **[HYPERPARAMETER_SEARCH.md](HYPERPARAMETER_SEARCH.md)** - Hyperparameter experiments guide
- **[EVALUATION_FIXES.md](EVALUATION_FIXES.md)** - Evaluation system fixes and determinism
- **[CODEBASE_ANALYSIS.md](CODEBASE_ANALYSIS.md)** - Comprehensive codebase analysis
- **[colab_setup.ipynb](colab_setup.ipynb)** - Google Colab setup notebook
- **[configs/README.md](configs/README.md)** - Configuration guide

---

## 🔍 How Evaluation Works (Detailed)

### Deterministic Evaluation Process

1. **Seed Initialization**:
   ```python
   # Set all random seeds globally
   np.random.seed(42)
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   torch.backends.cudnn.deterministic = True
   ```

2. **Test Prompt Loading**:
   - Loads 50 prompts (25 safe, 25 unsafe) from I2P dataset
   - Prompts shuffled deterministically with seed=42
   - Same prompts in same order every time

3. **For Each Prompt**:
   - **Original (Unsteered) Generation**:
     - Seed: `42 + prompt_index` (deterministic)
     - Intervention disabled (outside valid window)
     - Generates image without steering
     - Records ground truth label (not probe prediction)
   
   - **Steered Generation**:
     - Same seed: `42 + prompt_index` (deterministic)
     - Intervention enabled (window [5, 15])
     - Policy outputs steering actions
     - Generates image with steering
     - Records probe scores before/after steering

4. **Metric Calculation**:
   - **SSR**: Counts unsafe→safe conversions (using probe score improvements)
   - **FPR**: Counts safe→flagged conversions (using probe score increases)
   - **LPIPS**: Computes perceptual distance (deterministic)
   - **Transport Cost**: Sums squared action norms (deterministic)

5. **Result Saving**:
   - Saves metrics to JSON files
   - Saves sample image comparisons
   - All results are deterministic and reproducible

### Why Determinism Matters

- **Reproducibility**: Same evaluation → same results
- **Fair Comparison**: All policies evaluated on identical test set
- **Debugging**: Can identify issues by comparing runs
- **Scientific Rigor**: Results are verifiable and consistent

---

## ⚠️ Important Notes

- **Research Purpose**: This project is for academic research on AI safety and alignment
- **Model Selection**: Uses SD 1.4 (less censored) to study unsafe concept representation
- **Content Focus**: Nudity-only for clearer concept boundaries (not gore/violence)
- **Ethical Use**: All generated content is used solely for research and evaluation
- **Python Version**: Requires Python 3.11 for experiments (use `py -3.11` on Windows)
- **Determinism**: Always use `--seed 42` (or specify seed) for reproducible evaluations

---

## 📖 References

### Core Methodology

1. **Lamba, P., Ravish, K., Kushwaha, A., & Kumar, P. (2025).** "Alignment and Safety of Diffusion Models via Reinforcement Learning and Reward Modeling: A Survey." arXiv:2505.17352v1.  
   *Foundation for RL-based safety alignment in diffusion models.*

2. **Alain, G., & Bengio, Y. (2016).** "Understanding Intermediate Layers Using Linear Classifier Probes." arXiv:1610.01644.  
   *Methodology for linear probing to measure concept separability in latent spaces.*

3. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** "Proximal Policy Optimization Algorithms." arXiv:1707.06347.  
   *PPO algorithm used for policy training.*

4. **Wagenmaker, A., Nakamoto, M., Zhang, Y., et al. (2025).** "Steering Your Diffusion Policy with Latent Space Reinforcement Learning." arXiv:2506.15799v2.  
   *Latent space RL for diffusion model control.*

### Diffusion Models & ODE Formulation

5. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020).** "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.  
   *Probability flow ODE formulation for diffusion models.*

6. **Ho, J., Jain, A., & Abbeel, P. (2020).** "Denoising Diffusion Probabilistic Models." NeurIPS 2020.  
   *DDPM framework for generative modeling.*

7. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).** "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.  
   *Stable Diffusion architecture and latent space formulation.*

### Optimal Transport

8. **Villani, C. (2009).** "Optimal Transport: Old and New." Springer.  
   *Theoretical foundation for Wasserstein distance and optimal transport.*

9. **Peyré, G., & Cuturi, M. (2019).** "Computational Optimal Transport." Foundations and Trends in Machine Learning, 11(5-6), 355-607.  
   *Computational methods for optimal transport.*

### Evaluation Metrics

10. **Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017).** "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS 2017.  
    *Fréchet Inception Distance (FID) metric for image quality assessment.*

11. **Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).** "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR 2018.  
    *LPIPS (Learned Perceptual Image Patch Similarity) metric.*

### Layer Sensitivity & Concept Steering

12. **Lu, S., Wang, Z., et al. (2024).** "MACE: Mass Concept Erasure in Diffusion Models." CVPR 2024.  
    *Semantic Single Boundary (SSB) concept for optimal intervention timing.*

13. **Abbasian, M., Rajabzadeh, T., Moradipari, A., et al. (2023).** "Controlling the Latent Space of GANs through Reinforcement Learning: A Case Study on Task-based Image-to-Image Translation." arXiv:2307.13978v1.  
    *RL-based latent space control for generative models.*

### Datasets

14. **I2P Dataset:** AIML-TUDA/i2p on HuggingFace.  
    *Inappropriate Image Prompts benchmark for safety research.*

15. **Stable Diffusion 1.4:** CompVis/stable-diffusion-v1-4 on HuggingFace.  
    *Base diffusion model (OpenRAIL license).*

---

## 👥 Team

- **Alkan**
- **Durak**
- **Chiucchiolo**

---

## 📝 License

This project is for educational purposes as part of the Advanced Machine Learning course at Sapienza University of Rome.

---

*Last updated: December 2025*
