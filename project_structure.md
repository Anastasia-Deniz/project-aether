# Project Aether - Directory Structure

```
project-aether/
│
├── configs/                    # Hyperparameters and experiment configs
│   ├── base.yaml
│   ├── train_ppo.yaml
│   └── eval.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/                 # Core model components
│   │   ├── __init__.py
│   │   ├── base_model.py       # Wrapper for frozen diffusion model
│   │   ├── steering_policy.py  # The π_φ network
│   │   └── linear_probe.py     # Concept detection probes
│   │
│   ├── envs/                   # RL Environment
│   │   ├── __init__.py
│   │   └── diffusion_env.py    # Gymnasium env wrapping the ODE
│   │
│   ├── rewards/                # Reward computation
│   │   ├── __init__.py
│   │   ├── safety_reward.py    # R_safe from classifier
│   │   └── transport_reward.py # W2 cost computation
│   │
│   ├── training/               # Training loops
│   │   ├── __init__.py
│   │   ├── train_probe.py      # Phase 1: Linear probe training
│   │   ├── layer_sensitivity.py# Phase 1: SSB analysis
│   │   └── train_ppo.py        # Phase 2: Policy training
│   │
│   ├── evaluation/             # Metrics and benchmarks
│   │   ├── __init__.py
│   │   ├── metrics.py          # SSR, LPIPS, FPR computation
│   │   └── benchmark.py        # I2P and MS-COCO evaluation
│   │
│   └── utils/                  # Helpers
│       ├── __init__.py
│       ├── data.py             # Dataset loading
│       └── viz.py              # Trajectory visualization
│
├── data/
│   ├── i2p/                    # Unsafe benchmark (download separately)
│   └── mscoco_subset/          # Safe control group
│
├── checkpoints/                # Saved models
│   ├── probes/
│   └── policies/
│
├── outputs/                    # Generated images and logs
│   ├── images/
│   └── logs/
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_explore_latents.ipynb
│   ├── 02_probe_analysis.ipynb
│   └── 03_visualize_steering.ipynb
│
├── scripts/                    # Entry point scripts
│   ├── run_probe.py
│   ├── run_sensitivity.py
│   ├── run_train.py
│   └── run_eval.py
│
├── tests/                      # Unit tests
│   └── test_env.py
│
├── requirements.txt
├── setup.py
└── README.md
```

## Key Design Decisions

### 1. Gymnasium Environment
The diffusion process is wrapped as a Gymnasium environment:
- **State**: (z_t, t, probe_score)
- **Action**: Δz_t (steering vector)
- **Reward**: R_safe(x_0) - λ * ||a_t||²
- **Done**: When t reaches 0

### 2. Modular Rewards
Keep safety reward and transport cost separate. This lets you:
- Swap safety classifiers easily
- Tune λ without rewriting code
- Add new reward terms (e.g., CLIP similarity)

### 3. Config-Driven Experiments
Use YAML configs for all hyperparameters. This makes it easy to:
- Track experiments in W&B
- Reproduce results
- Run hyperparameter sweeps
