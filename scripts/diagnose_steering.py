"""
Diagnostic script to understand why steering has no visible effect.

This script will:
1. Check the actual action magnitudes being output by the policy
2. Check the max_action_norm setting
3. Check the transport cost penalty
4. Visualize action distributions
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.diffusion_env import DiffusionSteeringEnv, AetherConfig
from src.training.ppo_trainer import ActorCritic
from src.models.linear_probe import LinearProbe
from src.utils.data import I2PDataset


def diagnose_policy_actions(
    policy_path: str,
    probe_path: str,
    num_samples: int = 10,
    device: str = "cuda",
):
    """Diagnose what actions the policy is outputting."""
    
    print("="*60)
    print("STEERING DIAGNOSTIC")
    print("="*60)
    
    # Load config
    config = AetherConfig(
        model_id="CompVis/stable-diffusion-v1-4",
        num_inference_steps=20,
        device=device,
    )
    
    print(f"\nConfiguration:")
    print(f"  max_action_norm: {config.max_action_norm}")
    print(f"  lambda_transport: {config.lambda_transport}")
    print(f"  intervention_window: [{config.intervention_start}, {config.intervention_end}]")
    print(f"  steering_dim: {config.steering_dim}")
    
    # Load probe
    probe = LinearProbe(input_dim=config.latent_dim)
    probe_file = Path(probe_path) / "probe_t04.pt"
    if not probe_file.exists():
        probe_files = list(Path(probe_path).glob("probe_t*.pt"))
        if probe_files:
            probe_file = sorted(probe_files)[len(probe_files) // 2]
    probe.load_state_dict(torch.load(probe_file, map_location=device))
    probe = probe.to(device).eval()
    
    # Load policy
    checkpoint = torch.load(policy_path, map_location=device)
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    else:
        state_dict = checkpoint
    
    obs_dim = 258  # encoded_latent (256) + t (1) + score (1)
    action_dim = 256
    
    policy = ActorCritic(obs_dim, action_dim).to(device)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    
    # Create environment
    env = DiffusionSteeringEnv(config, linear_probe=probe)
    
    # Load test prompts
    dataset = I2PDataset()
    prompts = dataset.get_unsafe_prompts()[:num_samples]
    
    print(f"\nAnalyzing {len(prompts)} prompts...")
    
    all_action_norms = []
    all_projected_norms = []
    all_clipped_fractions = []
    probe_score_changes = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Diagnosing")):
        obs, _ = env.reset(seed=42 + i, options={'prompt': prompt})
        done = False
        step = 0
        
        episode_actions = []
        episode_projected = []
        episode_clipped = []
        initial_probe_score = None
        final_probe_score = None
        
        while not done:
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            action_np = action.cpu().numpy().squeeze()
            
            # Check if we're in intervention window
            should_intervene = (
                config.intervention_start <= step <= config.intervention_end
            )
            
            if should_intervene:
                # Measure action norm (before projection)
                action_norm = np.linalg.norm(action_np)
                episode_actions.append(action_norm)
                
                # Project action
                action_tensor = torch.from_numpy(action_np).to(device).to(config.dtype)
                with torch.no_grad():
                    delta_z = env.steering_proj(action_tensor)
                
                # Measure projected norm
                projected_norm = delta_z.norm().item()
                episode_projected.append(projected_norm)
                
                # Check if it would be clipped
                clipped = projected_norm > config.max_action_norm
                episode_clipped.append(clipped)
                
                if clipped:
                    # Measure actual applied norm (after clipping)
                    actual_norm = config.max_action_norm
                else:
                    actual_norm = projected_norm
                
                # Get probe score before steering
                if step == config.intervention_start:
                    with torch.no_grad():
                        z_flat = env.current_latent.flatten().float().unsqueeze(0).to(device)
                        initial_probe_score = probe(z_flat).item()
            
            # Step environment
            obs, _, done, _, info = env.step(action_np)
            step += 1
        
        # Get final probe score
        with torch.no_grad():
            final_latent = env.current_latent
            if final_latent is not None:
                z_flat = final_latent.flatten().float().unsqueeze(0).to(device)
                final_probe_score = probe(z_flat).item()
        
        if initial_probe_score is not None and final_probe_score is not None:
            probe_score_changes.append(final_probe_score - initial_probe_score)
        
        if episode_actions:
            all_action_norms.extend(episode_actions)
            all_projected_norms.extend(episode_projected)
            all_clipped_fractions.append(np.mean(episode_clipped))
    
    # Print statistics
    print("\n" + "="*60)
    print("ACTION STATISTICS")
    print("="*60)
    
    if all_action_norms:
        print(f"\nAction norms (before projection):")
        print(f"  Mean: {np.mean(all_action_norms):.6f}")
        print(f"  Std:  {np.std(all_action_norms):.6f}")
        print(f"  Min:  {np.min(all_action_norms):.6f}")
        print(f"  Max:  {np.max(all_action_norms):.6f}")
        print(f"  Median: {np.median(all_action_norms):.6f}")
    
    if all_projected_norms:
        print(f"\nProjected action norms (after projection, before clipping):")
        print(f"  Mean: {np.mean(all_projected_norms):.6f}")
        print(f"  Std:  {np.std(all_projected_norms):.6f}")
        print(f"  Min:  {np.min(all_projected_norms):.6f}")
        print(f"  Max:  {np.max(all_projected_norms):.6f}")
        print(f"  Median: {np.median(all_projected_norms):.6f}")
        print(f"\n  Max allowed (max_action_norm): {config.max_action_norm}")
        print(f"  Actions clipped: {np.mean(all_clipped_fractions)*100:.1f}%")
    
    if probe_score_changes:
        print(f"\nProbe score changes (final - initial):")
        print(f"  Mean: {np.mean(probe_score_changes):.6f}")
        print(f"  Std:  {np.std(probe_score_changes):.6f}")
        print(f"  Min:  {np.min(probe_score_changes):.6f}")
        print(f"  Max:  {np.max(probe_score_changes):.6f}")
        print(f"  (Negative = moved toward safety)")
    
    # Diagnose the problem
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if all_projected_norms:
        mean_projected = np.mean(all_projected_norms)
        max_projected = np.max(all_projected_norms)
        
        if mean_projected < config.max_action_norm * 0.1:
            print("❌ PROBLEM: Actions are TOO SMALL!")
            print(f"   Mean projected norm ({mean_projected:.6f}) is much smaller than max_action_norm ({config.max_action_norm})")
            print("   Possible causes:")
            print("   1. Transport cost penalty (lambda_transport) is too high")
            print("   2. Policy learned to be too conservative")
            print("   3. Reward shaping is not encouraging strong steering")
            print("\n   SOLUTION: Increase max_action_norm or decrease lambda_transport")
        
        elif max_projected > config.max_action_norm * 2:
            print("⚠️  WARNING: Many actions are being clipped!")
            print(f"   Max projected norm ({max_projected:.6f}) >> max_action_norm ({config.max_action_norm})")
            print("   This means the policy wants to steer more but is being limited")
            print("\n   SOLUTION: Increase max_action_norm")
        
        else:
            print("✓ Action magnitudes seem reasonable")
            print(f"   Mean: {mean_projected:.6f}, Max: {max_projected:.6f}, Limit: {config.max_action_norm}")
    
    if probe_score_changes:
        mean_change = np.mean(probe_score_changes)
        if mean_change > -0.01:
            print("\n❌ PROBLEM: Probe scores are NOT changing!")
            print(f"   Mean change: {mean_change:.6f} (should be negative for safety)")
            print("   This means steering is not affecting the latent representation")
            print("\n   SOLUTION: Check if actions are actually being applied")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("1. Try increasing max_action_norm (e.g., 0.1 → 0.5 or 1.0)")
    print("2. Try decreasing lambda_transport (e.g., 0.5 → 0.1 or 0.2)")
    print("3. Check if policy was trained long enough")
    print("4. Verify that intervention window is correct")
    print("5. Consider using gradient-based steering for comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose steering issues")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--probe_path", type=str, required=True, help="Path to probe directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    diagnose_policy_actions(
        policy_path=args.policy_path,
        probe_path=args.probe_path,
        num_samples=args.num_samples,
        device=args.device,
    )

