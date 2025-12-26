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
- **Empirical layer sensitivity analysis** with FID and SSR measurements ⭐ NEW
- **Layer sensitivity analysis** to identify optimal intervention points
- **Modular reward system** with separate safety (R_safe) and transport (W2) components
- **Optimal transport reward** combining safety and semantic alignment
- **Improved intermediate reward shaping** for faster learning ⭐ NEW
- **Configuration validation** to prevent runtime errors ⭐ NEW
- **Evaluation framework** with SSR, LPIPS, FID, and transport cost metrics

### Methodology

1. **Phase 1: Linear Probing** - Validate concept separability and identify optimal intervention timesteps
2. **Phase 2: Policy Training** - Train PPO policy to steer latents away from unsafe concepts
3. **Phase 3: Evaluation** - Measure steering effectiveness with comprehensive metrics

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Anastasia-Deniz/project-aether.git
cd project-aether

# Create environment
conda create -n aether python=3.11 -y
conda activate aether

# Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/test_setup.py
```

### Basic Usage

```bash
# Phase 1: Collect latents and train probes (nudity-focused)
python scripts/collect_latents.py \
    --num_samples 50 \
    --num_steps 20 \
    --focus_nudity \
    --hard_only \
    --min_nudity_pct 50.0 \
    --device cuda

python scripts/train_probes.py --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS

# (Optional) Phase 1.5: Measure empirical layer sensitivity for better accuracy
python scripts/measure_layer_sensitivity.py \
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS \
    --num_samples 20 \
    --device cuda

# Re-train probes with empirical measurements
python scripts/train_probes.py \
    --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS \
    --use_empirical

# Phase 2: Train PPO policy
python scripts/train_ppo.py --config configs/train_ppo_best.yaml

# Phase 3: Evaluate policy
python scripts/evaluate_ppo.py \
    --policy_path outputs/ppo/aether_ppo_YYYYMMDD_HHMMSS/final_policy.pt \
    --probe_path checkpoints/probes/run_YYYYMMDD_HHMMSS/pytorch/ \
    --num_samples 50
```

**For detailed instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

---

## 📋 Requirements

### Minimum
- Python 3.10 or 3.11
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
│   ├── models/             # Linear probes (linear_probe.py)
│   ├── rewards/            # Reward computation (safety, transport)
│   ├── training/           # PPO trainer (ppo_trainer.py)
│   ├── evaluation/         # Evaluation metrics (SSR, LPIPS, etc.)
│   └── utils/              # Data loading utilities
├── scripts/                # Executable scripts
│   ├── collect_latents.py      # Phase 1: Collect latents
│   ├── train_probes.py         # Phase 1: Train probes
│   ├── measure_layer_sensitivity.py  # Phase 1: Empirical FID/SSR measurement ⭐ NEW
│   ├── train_ppo.py            # Phase 2: Train policy
│   ├── evaluate_ppo.py         # Phase 3: Evaluate policy
│   ├── generate_images_from_latents.py  # Visualize generated images
│   └── visualize_probe_results.py        # Verify probe accuracy
├── configs/                # Configuration files
│   ├── train_ppo_best.yaml     # Recommended config
│   ├── rtx4050_optimized.yaml  # For 6GB GPUs
│   └── colab_optimized.yaml    # For Google Colab
├── tests/                  # Unit tests
├── data/                   # Data directory (latents, cache)
├── checkpoints/            # Saved models (probes, policies)
└── outputs/                # Generated outputs (evaluation, visualizations)
```

---

## 🔬 Key Features

### Content Focus
- **Nudity-only focus** for clearer concept boundaries
- **Strict filtering** (≥50% nudity, ≥60% inappropriate, hard prompts only)
- **Model selection:** SD 1.4 (less censored) for research purposes

### Visualization & Verification
- **Image generation** from collected latents
- **HTML viewer** for browsing safe/unsafe images
- **Probe visualization** with confusion matrices and probability distributions
- **Comprehensive metrics** (SSR, FPR, LPIPS, Transport Cost)

### Memory Optimizations
- Latent encoder reduces observation space (16K → 256 dims)
- CUDA cache clearing
- Trajectory storage disabled during training
- Optimized configs for 6GB GPUs

---

## 📊 Results

### Phase 1: Linear Probing
- ✅ **90% accuracy** at timesteps 2-3
- ✅ **97% AUC** at timestep 1
- ✅ Linear separability confirmed
- ✅ Optimal intervention window: [2, 6] (for 8-step) or [5, 15] (for 20-step)

### Phase 2: PPO Training
- ✅ Training completed successfully
- ✅ Policy loss decreased (learning confirmed)
- ✅ Memory optimizations enable training on 6GB GPUs

### Phase 3: Evaluation
- Initial results: SSR 13.3%, FPR 26.7%, LPIPS 0.32
- Framework ready for extended training

---

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup and usage instructions
- **[colab_setup.ipynb](colab_setup.ipynb)** - Google Colab setup notebook
- **[kaggle_setup.ipynb](kaggle_setup.ipynb)** - Kaggle notebook setup ⭐ NEW
- **[configs/README.md](configs/README.md)** - Configuration guide
- **[FIXES_APPLIED.md](FIXES_APPLIED.md)** - Recent improvements and fixes
- **[CODEBASE_ANALYSIS.md](CODEBASE_ANALYSIS.md)** - Comprehensive codebase analysis

---

## 🔧 Configuration

### Recommended Configs

| Config | Use Case | Description |
|--------|----------|-------------|
| `train_ppo_best.yaml` | **Recommended** | Optimal from experiments (λ=0.5, 100K steps) |
| `train_ppo_optimized.yaml` | Extended training | Maximum performance (λ=0.8, 200K steps) |
| `rtx4050_optimized.yaml` | Low VRAM | Optimized for 6GB GPUs |
| `colab_optimized.yaml` | Google Colab/Kaggle | Optimized for T4/P100 GPU (16GB VRAM) |

---

## 🧪 Testing

```bash
# Run unit tests
python tests/test_env.py
python tests/test_rewards.py
python tests/test_ppo.py

# Quick component test
python scripts/quick_test.py
```

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

## ⚠️ Important Notes

- **Research Purpose:** This project is for academic research on AI safety and alignment
- **Model Selection:** Uses SD 1.4 (less censored) to study unsafe concept representation
- **Content Focus:** Nudity-only for clearer concept boundaries (not gore/violence)
- **Ethical Use:** All generated content is used solely for research and evaluation

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
