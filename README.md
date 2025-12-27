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
- **Linear probing** for concept detection in latent space
- **Empirical layer sensitivity analysis** with FID and SSR measurements
- **Layer sensitivity analysis** to identify optimal intervention points
- **Modular reward system** with separate safety (R_safe) and transport cost components
- **Optimal transport-inspired reward** combining safety and semantic alignment (transport cost: Σ||Δz_t||²)
- **Configuration validation** to prevent runtime errors
- **Hyperparameter experiment framework** for systematic optimization
- **Deterministic evaluation framework** with SSR, LPIPS, FPR, and transport cost metrics
- **Robust evaluation system** with fixed seeds and statistical validation

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


### Phase 3: Evaluation

**Goal**: Measure steering effectiveness with comprehensive, deterministic metrics following academic standards.

**Steps**:

1. **Robust Policy Evaluation** (Recommended: `scripts/evaluate_ppo_robust.py`)
   - Validates probe accuracy before evaluation
   - Loads trained policy and probe
   - Generates images with and without steering on test prompts
   - Computes metrics: SSR, FPR, LPIPS, Transport Cost
   - **Statistical robustness**: Confidence intervals (95% CI) for continuous metrics
   - **Deterministic**: Uses fixed seed (default: 42) for reproducibility
   - **Ground truth**: Uses dataset labels, not probe predictions
   - Output: Results saved to `outputs/evaluation/eval_robust_YYYYMMDD_HHMMSS/`
   - See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed methodology

2. **Standard Policy Evaluation** (`scripts/evaluate_ppo.py`)
   - Basic evaluation without probe validation
   - Same metrics as robust version
   - Use for quick checks or when probe is already validated

3. **Batch Evaluation** (Recommended: `scripts/evaluate_all_policies_robust.py`)
   - Automatically finds all trained policies
   - Evaluates each using robust evaluation (with probe validation)
   - Compares results with confidence intervals
   - Generates comprehensive comparison report
   - Output: Summary saved to `outputs/policies_comparison_robust.json`
   
   Standard version: `scripts/evaluate_all_policies.py` (uses basic evaluation)

4. **Hyperparameter Search** (`scripts/hyperparameter_search.py`)
   - Runs training + evaluation for multiple experiment configs
   - Compares results to find best hyperparameters
   - Output: Ranked list of experiments with metrics

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

# Phase 3: Evaluate policy (ROBUST, RECOMMENDED)
py -3.11 scripts/evaluate_ppo_robust.py \
    --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt \
    --probe_path checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ \
    --num_samples 100 \
    --seed 42  # Fixed seed for reproducibility

# Or use standard evaluation (faster, less validation)
py -3.11 scripts/evaluate_ppo.py \
    --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt \
    --probe_path checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ \
    --num_samples 100 \
    --seed 42

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

---

## Requirements

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

## Project Structure

```
project-aether/
├── src/                    # Core source code
│   ├── envs/              # RL environment (diffusion_env.py)
│   │   └── diffusion_env.py  # Gymnasium environment wrapping Stable Diffusion
│   ├── models/             # Linear probes (linear_probe.py)
│   ├── rewards/            # Reward computation
│   │   ├── safety_reward.py    # Safety reward (probe-based or classifier-based)
│   │   └── transport_reward.py # Transport cost (Wasserstein-2 inspired)
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

## Key Features

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

**Evaluation Process**:
1. Loads test prompts (50 samples: 25 safe, 25 unsafe)
2. For each prompt:
   - Generates image **without** steering (baseline, seed=42+i)
   - Generates image **with** steering (policy, same seed=42+i)
   - Compares probe scores before/after steering
3. Computes metrics deterministically
4. Saves results to `outputs/evaluation/eval_robust_YYYYMMDD_HHMMSS/` (with confidence intervals and probe validation)

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
   - **SSR**: Counts unsafe→safe conversions. Uses ground truth labels to identify originally unsafe samples, then checks if probe prediction after steering indicates safe (0). Formula: (unsafe samples that became safe) / (total unsafe samples)
   - **FPR**: Counts safe→flagged conversions. Uses ground truth labels to identify originally safe samples, then checks if probe prediction after steering indicates unsafe (1). Formula: (safe samples flagged as unsafe) / (total safe samples)
   - **LPIPS**: Computes perceptual distance (deterministic)
   - **Transport Cost**: Sums squared action norms (Wasserstein-2 inspired cost, deterministic)

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

### Optimal Transport (Inspiration)

8. **Villani, C. (2009).** "Optimal Transport: Old and New." Springer.  
   *Theoretical foundation for Wasserstein distance and optimal transport. Note: Our transport cost Σ||Δz_t||² is inspired by Wasserstein-2 distance but is a simplified proxy that measures the total squared displacement of steering actions.*

9. **Peyré, G., & Cuturi, M. (2019).** "Computational Optimal Transport." Foundations and Trends in Machine Learning, 11(5-6), 355-607.  
   *Computational methods for optimal transport. Our implementation uses the sum of squared action norms as a computationally efficient proxy for the full Wasserstein-2 distance between distributions.*

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
