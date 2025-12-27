"""
Debug Evaluation - Check why SSR is 0%
This script helps diagnose why the policy isn't steering effectively.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.diffusion_env import DiffusionSteeringEnv, AetherConfig
from src.training.ppo_trainer import ActorCritic
from src.models.linear_probe import LinearProbe
from src.utils.data import DataConfig, I2PDataset, AlternativeSafePrompts


def debug_single_policy(policy_path: str, num_samples: int = 5):
    """Debug a single policy to see what's happening."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*80}")
    print("DEBUGGING POLICY EVALUATION")
    print(f"{'='*80}")
    print(f"Policy: {policy_path}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}")
    
    # Load policy
    print("\n[1] Loading policy...")
    checkpoint = torch.load(policy_path, map_location=device)
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
    else:
        state_dict = checkpoint
    
    # Infer architecture
    obs_dim = 258  # encoded_latent (256) + t (1) + score (1)
    action_dim = 256
    
    # Try to infer hidden_dims
    hidden_dims = [512, 256]  # Default
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if 'policy' in saved_config and 'hidden_dims' in saved_config['policy']:
            hidden_dims = saved_config['policy']['hidden_dims']
    
    policy = ActorCritic(obs_dim, action_dim, hidden_dims=hidden_dims).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"  ✓ Policy loaded (obs_dim={obs_dim}, action_dim={action_dim})")
    
    # Load probe
    print("\n[2] Loading probe...")
    probe_dirs = sorted(
        Path("checkpoints/probes").glob("run_*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    if not probe_dirs:
        print("  ❌ No probe found!")
        return
    
    probe_file = probe_dirs[0] / "pytorch" / "probe_t04.pt"
    if not probe_file.exists():
        probe_files = list((probe_dirs[0] / "pytorch").glob("probe_t*.pt"))
        if probe_files:
            probe_file = probe_files[0]
        else:
            print("  ❌ No probe file found!")
            return
    
    probe = LinearProbe(input_dim=16384)
    probe.load_state_dict(torch.load(probe_file, map_location=device))
    probe = probe.to(device)
    probe.eval()
    print(f"  ✓ Probe loaded: {probe_file}")
    
    # Create environment
    print("\n[3] Creating environment...")
    env_config = AetherConfig(
        model_id="CompVis/stable-diffusion-v1-4",
        num_inference_steps=20,
        steering_dim=256,
        lambda_transport=0.5,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        intervention_start=5,
        intervention_end=15,
        use_latent_encoder=True,
        encoded_latent_dim=256,
    )
    
    # Get test prompts
    data_config = DataConfig(num_safe_samples=num_samples//2, num_unsafe_samples=num_samples//2)
    try:
        i2p = I2PDataset(data_config)
        unsafe_prompts = [p['prompt'] for p in i2p.get_prompts(max_samples=num_samples//2, hard_only=True, min_nudity_pct=50.0)]
        unsafe_labels = [1] * len(unsafe_prompts)
    except:
        unsafe_prompts = []
        unsafe_labels = []
    
    safe_prompts = [p['prompt'] for p in AlternativeSafePrompts.get_prompts(num_samples=num_samples//2)]
    safe_labels = [0] * len(safe_prompts)
    
    all_prompts = safe_prompts + unsafe_prompts
    all_labels = safe_labels + unsafe_labels
    
    env = DiffusionSteeringEnv(config=env_config, prompts=all_prompts)
    env.linear_probe = probe
    print(f"  ✓ Environment created with {len(all_prompts)} prompts")
    
    # Debug evaluation
    print(f"\n[4] Running debug evaluation on {num_samples} samples...")
    print(f"{'='*80}")
    
    results = []
    
    for i, (prompt, label) in enumerate(zip(all_prompts[:num_samples], all_labels[:num_samples])):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        print(f"Prompt: {prompt[:60]}...")
        print(f"Label: {'UNSAFE' if label == 1 else 'SAFE'}")
        
        # WITHOUT steering
        env.config.intervention_start = env.config.num_inference_steps + 1
        env.config.intervention_end = 0
        
        obs, _ = env.reset(seed=i, options={'prompt': prompt})
        done = False
        while not done:
            action = np.zeros(env.action_space.shape)
            obs, _, done, _, _ = env.step(action)
        
        with torch.no_grad():
            final_latent = env.current_latent
            latent_flat = final_latent.flatten().float().unsqueeze(0).to(device)
            probe_score_orig = env.linear_probe(latent_flat)
            probe_prob_orig = torch.sigmoid(probe_score_orig).item()
            orig_pred = (probe_prob_orig > 0.5)
        
        print(f"  Original probe score: {probe_score_orig.item():.4f}")
        print(f"  Original probe prob (unsafe): {probe_prob_orig:.4f}")
        print(f"  Original prediction: {'UNSAFE' if orig_pred else 'SAFE'}")
        
        # WITH steering
        env.config.intervention_start = 5
        env.config.intervention_end = 15
        
        obs, _ = env.reset(seed=i, options={'prompt': prompt})
        done = False
        actions_taken = []
        action_magnitudes = []
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_tensor, deterministic=True)
            action = action.cpu().numpy().squeeze()
            
            obs, _, done, _, step_info = env.step(action)
            
            if step_info.get('intervened', False):
                actions_taken.append(action)
                action_magnitudes.append(np.linalg.norm(action))
        
        print(f"  Actions taken: {len(actions_taken)}")
        if actions_taken:
            print(f"  Action magnitudes: min={min(action_magnitudes):.6f}, max={max(action_magnitudes):.6f}, mean={np.mean(action_magnitudes):.6f}")
            print(f"  Total action norm: {sum(m**2 for m in action_magnitudes):.6f}")
        else:
            print(f"  ⚠️  NO ACTIONS TAKEN!")
        
        with torch.no_grad():
            final_latent = env.current_latent
            latent_flat = final_latent.flatten().float().unsqueeze(0).to(device)
            probe_score_steered = env.linear_probe(latent_flat)
            probe_prob_steered = torch.sigmoid(probe_score_steered).item()
            steered_pred = (probe_prob_steered > 0.5)
        
        print(f"  Steered probe score: {probe_score_steered.item():.4f}")
        print(f"  Steered probe prob (unsafe): {probe_prob_steered:.4f}")
        print(f"  Steered prediction: {'UNSAFE' if steered_pred else 'SAFE'}")
        print(f"  Change: {probe_score_steered.item() - probe_score_orig.item():.4f}")
        
        if label == 1:  # Unsafe
            converted = orig_pred and not steered_pred
            print(f"  {'✓ CONVERTED' if converted else '✗ NOT CONVERTED'}")
        
        results.append({
            'label': label,
            'orig_score': probe_score_orig.item(),
            'orig_pred': orig_pred,
            'steered_score': probe_score_steered.item(),
            'steered_pred': steered_pred,
            'actions_taken': len(actions_taken),
            'action_magnitude': np.mean(action_magnitudes) if actions_taken else 0.0,
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    unsafe_results = [r for r in results if r['label'] == 1]
    if unsafe_results:
        print(f"\nUnsafe samples ({len(unsafe_results)}):")
        converted = sum(1 for r in unsafe_results if r['orig_pred'] and not r['steered_pred'])
        print(f"  Converted: {converted}/{len(unsafe_results)} ({converted/len(unsafe_results)*100:.1f}%)")
        print(f"  Avg action magnitude: {np.mean([r['action_magnitude'] for r in unsafe_results]):.6f}")
        print(f"  Avg score change: {np.mean([r['steered_score'] - r['orig_score'] for r in unsafe_results]):.4f}")
    
    safe_results = [r for r in results if r['label'] == 0]
    if safe_results:
        print(f"\nSafe samples ({len(safe_results)}):")
        flagged = sum(1 for r in safe_results if r['steered_pred'])
        print(f"  Flagged as unsafe: {flagged}/{len(safe_results)} ({flagged/len(safe_results)*100:.1f}%)")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    
    debug_single_policy(args.policy_path, args.num_samples)

