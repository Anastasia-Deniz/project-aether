"""
Quick evaluation script that overrides max_action_norm to test if larger actions help.

This allows you to test existing policies with larger action norms without retraining.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the robust evaluation function
from scripts.evaluate_ppo_robust import (
    load_policy_robust,
    load_probe_robust,
    evaluate_policy_robust,
    main as robust_main,
)
from src.envs.diffusion_env import DiffusionSteeringEnv, AetherConfig
from src.utils.data import I2PDataset
from src.evaluation.metrics import EvaluationMetrics
import torch
import numpy as np


def evaluate_with_override(
    policy_path: str,
    probe_path: str,
    max_action_norm: float = 1.0,
    lambda_transport: float = None,
    num_samples: int = 50,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Evaluate policy with overridden max_action_norm.
    
    This allows testing if larger actions would help without retraining.
    """
    print("="*60)
    print("EVALUATION WITH ACTION NORM OVERRIDE")
    print("="*60)
    print(f"max_action_norm: {max_action_norm} (overridden)")
    if lambda_transport is not None:
        print(f"lambda_transport: {lambda_transport} (overridden)")
    print("="*60)
    
    # Create config with overrides
    config = AetherConfig(
        model_id="CompVis/stable-diffusion-v1-4",
        num_inference_steps=20,
        device=device,
        max_action_norm=max_action_norm,  # OVERRIDE
    )
    
    if lambda_transport is not None:
        config.lambda_transport = lambda_transport  # OVERRIDE
    
    # Load probe
    probe = load_probe_robust(probe_path, latent_dim=config.latent_dim, device=device)
    if probe is None:
        print("ERROR: Could not load probe")
        return None
    
    # Load policy
    obs_dim = config.observation_dim
    action_dim = config.steering_dim
    policy = load_policy_robust(policy_path, obs_dim, action_dim, device)
    
    # Create environment
    env = DiffusionSteeringEnv(config, linear_probe=probe)
    
    # Load test data
    dataset = I2PDataset()
    prompts = dataset.get_test_prompts(num_samples=num_samples)
    labels = dataset.get_test_labels(num_samples=num_samples)
    
    # Evaluate
    metrics, diagnostics = evaluate_policy_robust(
        env=env,
        policy=policy,
        probe=probe,
        prompts=prompts,
        labels=labels,
        device=device,
        num_samples=num_samples,
        intervention_start=config.intervention_start,
        intervention_end=config.intervention_end,
        seed=seed,
    )
    
    print("\n" + "="*60)
    print("RESULTS WITH OVERRIDE")
    print("="*60)
    print(f"SSR: {metrics.ssr:.4f} (higher is better)")
    print(f"FPR: {metrics.fpr:.4f} (lower is better)")
    print(f"LPIPS: {metrics.lpips_mean:.4f} ± {metrics.lpips_std:.4f} (lower is better)")
    print(f"Transport Cost: {metrics.transport_cost_mean:.4f} ± {metrics.transport_cost_std:.4f}")
    print("="*60)
    
    return metrics, diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate policy with overridden max_action_norm"
    )
    parser.add_argument("--policy_path", type=str, required=True, help="Path to policy")
    parser.add_argument("--probe_path", type=str, required=True, help="Path to probe directory")
    parser.add_argument("--max_action_norm", type=float, default=1.0, 
                       help="Override max_action_norm (default: 1.0)")
    parser.add_argument("--lambda_transport", type=float, default=None,
                       help="Override lambda_transport (optional)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    evaluate_with_override(
        policy_path=args.policy_path,
        probe_path=args.probe_path,
        max_action_norm=args.max_action_norm,
        lambda_transport=args.lambda_transport,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
    )

