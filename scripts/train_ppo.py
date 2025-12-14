"""
Project Aether - PPO Training Script
Phase 2: Train the steering policy using PPO.

This script:
1. Loads the diffusion environment with optional pre-trained probe
2. Configures PPO hyperparameters from YAML or command line
3. Trains the steering policy
4. Saves checkpoints and training curves

Usage:
    python scripts/train_ppo.py --config configs/train_ppo.yaml
    
    # Quick test run
    python scripts/train_ppo.py --quick
    
    # With custom settings
    python scripts/train_ppo.py \
        --total_timesteps 100000 \
        --lambda_transport 0.5 \
        --probe_path ./checkpoints/probes/latest/pytorch/
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.diffusion_env import DiffusionSteeringEnv, AetherConfig
from src.training.ppo_trainer import AetherPPOTrainer, PPOConfig
from src.utils.data import DataConfig, I2PDataset, AlternativeSafePrompts


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO steering policy")
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    # Quick test mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with minimal settings"
    )
    
    # Environment settings
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",  # SD 1.4 - less censored
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,  # SD 1.4 uses 20-50 steps
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--lambda_transport",
        type=float,
        default=0.5,
        help="Transport cost penalty coefficient"
    )
    parser.add_argument(
        "--steering_dim",
        type=int,
        default=256,
        help="Steering vector dimension"
    )
    
    # Intervention window (adjusted for SD 1.4's 20 steps)
    parser.add_argument(
        "--intervention_start",
        type=int,
        default=5,  # ~25% of generation (scaled from 2/8 to 5/20)
        help="Start step for intervention"
    )
    parser.add_argument(
        "--intervention_end",
        type=int,
        default=15,  # ~75% of generation (scaled from 6/8 to 15/20)
        help="End step for intervention"
    )
    
    # PPO settings
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=500000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Steps per rollout"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    # Probe settings
    parser.add_argument(
        "--probe_path",
        type=str,
        default=None,
        help="Path to trained linear probe"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/ppo",
        help="Output directory"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="aether_ppo",
        help="Experiment name for logging"
    )
    
    # Logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detected if not specified)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def truncate_prompt(prompt: str, max_words: int = 50) -> str:
    """Truncate prompt to avoid CLIP's 77 token limit."""
    words = prompt.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return prompt


def get_prompts(num_prompts: int = 100, seed: int = 42) -> list:
    """Load prompts for training."""
    # Mix of safe and unsafe prompts
    data_config = DataConfig(
        num_safe_samples=num_prompts // 2,
        num_unsafe_samples=num_prompts // 2,
        random_seed=seed,
    )
    
    # Load unsafe prompts
    try:
        i2p = I2PDataset(data_config)
        unsafe_prompts = i2p.get_prompts(max_samples=num_prompts // 2)
    except Exception as e:
        print(f"Warning: Could not load I2P dataset: {e}", flush=True)
        unsafe_prompts = []
    
    # Load safe prompts
    safe_prompts = AlternativeSafePrompts.get_prompts(
        num_samples=num_prompts // 2,
        seed=seed,
    )
    
    # Combine and extract just the prompt text
    # Truncate prompts to avoid CLIP's 77 token limit
    all_prompts = [truncate_prompt(p['prompt']) for p in safe_prompts]
    if unsafe_prompts:
        all_prompts.extend([truncate_prompt(p['prompt']) for p in unsafe_prompts])
    
    np.random.seed(seed)
    np.random.shuffle(all_prompts)
    
    print(f"Prompts truncated to max 50 words to avoid CLIP token limit", flush=True)
    return all_prompts


def main():
    import sys
    args = parse_args()
    
    # Load config from file if provided
    if args.config:
        file_config = load_config(args.config)
    else:
        file_config = {}
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"{'='*60}", flush=True)
    print("AETHER PPO TRAINING", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Device: {device}", flush=True)
    
    # Quick test mode
    if args.quick:
        print("\n⚡ QUICK TEST MODE ⚡")
        args.total_timesteps = 200
        args.n_steps = 20
        args.num_inference_steps = 5
        args.batch_size = 4
    
    # Environment configuration
    env_config = AetherConfig(
        model_id=file_config.get('env', {}).get('model_id', args.model_id),
        num_inference_steps=file_config.get('env', {}).get('num_inference_steps', args.num_inference_steps),
        guidance_scale=file_config.get('env', {}).get('guidance_scale', 7.5),
        steering_dim=file_config.get('env', {}).get('steering_dim', args.steering_dim),
        lambda_transport=file_config.get('reward', {}).get('lambda_transport', args.lambda_transport),
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        intervention_start=file_config.get('env', {}).get('intervention_start', args.intervention_start),
        intervention_end=file_config.get('env', {}).get('intervention_end', args.intervention_end),
        use_latent_encoder=True,
        encoded_latent_dim=256,
    )
    
    print(f"\nEnvironment config:", flush=True)
    print(f"  Model: {env_config.model_id}", flush=True)
    print(f"  Inference steps: {env_config.num_inference_steps}", flush=True)
    print(f"  Steering dim: {env_config.steering_dim}", flush=True)
    print(f"  Lambda transport: {env_config.lambda_transport}", flush=True)
    print(f"  Intervention window: [{env_config.intervention_start}, {env_config.intervention_end}]", flush=True)
    print(f"  Use latent encoder: {env_config.use_latent_encoder}", flush=True)
    print(f"  Observation dim: {env_config.observation_dim}", flush=True)
    sys.stdout.flush()
    
    # Load prompts
    print("\nLoading prompts...", flush=True)
    prompts = get_prompts(num_prompts=100, seed=42)
    print(f"Loaded {len(prompts)} prompts", flush=True)
    
    # Create environment
    print("\nCreating environment...", flush=True)
    sys.stdout.flush()
    
    # For quick test without GPU, skip model loading
    if args.quick and device == "cpu":
        print("Quick test on CPU - creating mock environment...")
        env = DiffusionSteeringEnv(
            config=env_config,
            prompts=prompts,
            load_model=False,  # Don't load the big model
        )
        print("Note: Running in test mode without full model")
    else:
        env = DiffusionSteeringEnv(
            config=env_config,
            prompts=prompts,
        )
    
    print(f"Observation space: {env.observation_space}", flush=True)
    print(f"Action space: {env.action_space}", flush=True)
    sys.stdout.flush()
    
    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=file_config.get('ppo', {}).get('learning_rate', args.learning_rate),
        n_steps=file_config.get('ppo', {}).get('n_steps', args.n_steps),
        batch_size=file_config.get('ppo', {}).get('batch_size', args.batch_size),
        n_epochs=file_config.get('ppo', {}).get('n_epochs', 10),
        gamma=file_config.get('ppo', {}).get('gamma', 0.99),
        gae_lambda=file_config.get('ppo', {}).get('gae_lambda', 0.95),
        clip_range=file_config.get('ppo', {}).get('clip_range', 0.2),
        vf_coef=file_config.get('ppo', {}).get('vf_coef', 0.5),
        ent_coef=file_config.get('ppo', {}).get('ent_coef', 0.01),
        max_grad_norm=file_config.get('ppo', {}).get('max_grad_norm', 0.5),
        total_timesteps=file_config.get('ppo', {}).get('total_timesteps', args.total_timesteps),
        hidden_dims=file_config.get('policy', {}).get('hidden_dims', [512, 256]),
        device=device,
    )
    
    print(f"\nPPO config:", flush=True)
    print(f"  Total timesteps: {ppo_config.total_timesteps:,}", flush=True)
    print(f"  Steps per rollout: {ppo_config.n_steps}", flush=True)
    print(f"  Batch size: {ppo_config.batch_size}", flush=True)
    print(f"  Learning rate: {ppo_config.learning_rate}", flush=True)
    sys.stdout.flush()
    
    # Probe path - auto-detect if not provided or path doesn't exist
    probe_path = file_config.get('reward', {}).get('probe_path', args.probe_path)
    
    # Auto-detect latest probe if path is None, empty, or doesn't exist
    if not probe_path or probe_path == "auto" or (probe_path and not Path(probe_path).exists()):
        print("Auto-detecting latest probe...")
        probe_dirs = sorted(Path("checkpoints/probes").glob("run_*"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        if probe_dirs:
            latest_probe_dir = probe_dirs[-1] / "pytorch"
            if latest_probe_dir.exists():
                probe_path = str(latest_probe_dir)
                print(f"Found latest probe: {probe_path}")
            else:
                print(f"Warning: Probe directory {latest_probe_dir} not found. Training without probe.")
                probe_path = None
        else:
            print("Warning: No probe directories found. Training without probe.")
            probe_path = None
    
    # Create trainer
    trainer = AetherPPOTrainer(
        env=env,
        config=ppo_config,
        probe_path=probe_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb,
    )
    
    # Train
    print(f"\nStarting training...", flush=True)
    sys.stdout.flush()
    history = trainer.train(total_timesteps=ppo_config.total_timesteps)
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(history['rewards'])
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Update')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history['policy_loss'])
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history['value_loss'])
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history['entropy'])
        axes[1, 1].set_title('Entropy')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(trainer.run_dir / "training_curves.png", dpi=150)
        print(f"\nSaved training curves to {trainer.run_dir / 'training_curves.png'}")
        
    except Exception as e:
        print(f"Could not save plots: {e}")
    
    print(f"\n✓ Training complete!", flush=True)
    print(f"Output: {trainer.run_dir}", flush=True)
    
    return trainer.run_dir


if __name__ == "__main__":
    main()

