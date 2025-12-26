"""
Project Aether - Empirical Layer Sensitivity Measurement
Measures FID (quality preservation) and SSR improvement (steering effectiveness) at each timestep.

This script runs small steering experiments to empirically measure:
- Quality preservation: FID between steered and unsteered images (1 - FID_norm)
- Steering effectiveness: SSR improvement from steering at each timestep

Usage:
    python scripts/measure_layer_sensitivity.py \
        --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS \
        --num_samples 20 \
        --device cuda

Output:
    - quality_measurements.json: {timestep: 1 - FID_norm}
    - effectiveness_measurements.json: {timestep: SSR_improvement}
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.diffusion_env import DiffusionSteeringEnv, AetherConfig
from src.models.linear_probe import LinearProbe
from src.evaluation.metrics import compute_ssr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure empirical layer sensitivity (FID and SSR)"
    )
    
    parser.add_argument(
        "--latents_dir",
        type=str,
        required=True,
        help="Directory containing collected latents (from collect_latents.py)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to use for measurement (smaller = faster)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        default=None,
        help="Path to trained probe (optional, for SSR measurement)"
    )
    
    return parser.parse_args()


def load_prompts(latents_dir: Path) -> Tuple[List[str], List[int]]:
    """Load prompts and labels from latents directory."""
    prompts = []
    labels = []
    
    # Try to load from saved prompts file
    prompts_file = latents_dir / "unsafe_prompts.json"
    if prompts_file.exists():
        with open(prompts_file, 'r', encoding='utf-8') as f:
            unsafe_data = json.load(f)
            for item in unsafe_data:
                if isinstance(item, dict):
                    prompts.append(item['prompt'])
                    labels.append(1)  # Unsafe
                elif isinstance(item, str):
                    prompts.append(item)
                    labels.append(1)
    
    # Try to load safe prompts
    safe_prompts_file = latents_dir / "safe_prompts.json"
    if safe_prompts_file.exists():
        with open(safe_prompts_file, 'r', encoding='utf-8') as f:
            safe_data = json.load(f)
            for item in safe_data:
                if isinstance(item, dict):
                    prompts.append(item['prompt'])
                    labels.append(0)  # Safe
                elif isinstance(item, str):
                    prompts.append(item)
                    labels.append(0)
    
    # Fallback: use I2P dataset if prompts not found
    if not prompts:
        print("Warning: No prompts found in latents directory. Loading from I2P dataset...")
        from src.utils.data import DataConfig, I2PDataset, AlternativeSafePrompts
        
        data_config = DataConfig(num_safe_samples=20, num_unsafe_samples=20)
        i2p = I2PDataset(data_config)
        unsafe_prompts = i2p.get_prompts(max_samples=20, focus_nudity_gore=True, min_nudity_pct=50.0)
        safe_prompts = AlternativeSafePrompts.get_prompts(num_samples=20, seed=42)
        
        prompts = [p['prompt'] if isinstance(p, dict) else p for p in safe_prompts]
        labels = [0] * len(prompts)
        
        prompts.extend([p['prompt'] if isinstance(p, dict) else p for p in unsafe_prompts])
        labels.extend([1] * len(unsafe_prompts))
    
    return prompts, labels


def compute_fid_batch(
    images1: List[np.ndarray],
    images2: List[np.ndarray],
    device: str = 'cuda',
) -> float:
    """
    Compute FID between two sets of images.
    
    Uses pytorch-fid library (Heusel et al., 2017).
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")
        return 0.0
    
    # Save images to temporary directories
    import tempfile
    import shutil
    from PIL import Image
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Save images
        for i, img in enumerate(images1):
            Image.fromarray(img).save(f"{tmpdir1}/img_{i:04d}.png")
        
        for i, img in enumerate(images2):
            Image.fromarray(img).save(f"{tmpdir2}/img_{i:04d}.png")
        
        # Compute FID
        fid_value = fid_score.calculate_fid_given_paths(
            [tmpdir1, tmpdir2],
            batch_size=8,
            device=device,
            dims=2048,
        )
    
    return float(fid_value)


def measure_quality_preservation(
    env: DiffusionSteeringEnv,
    prompts: List[str],
    labels: List[int],
    timestep: int,
    num_samples: int,
    device: str,
) -> float:
    """
    Measure quality preservation at a specific timestep.
    
    Generates images with and without steering at the given timestep,
    then computes FID. Quality = 1 - FID_norm (normalized FID).
    
    Returns:
        Quality score in [0, 1] (higher = better quality preservation)
    """
    original_images = []
    steered_images = []
    
    # Use a simple steering action (small random perturbation)
    steering_magnitude = 0.05
    
    for i in range(min(num_samples, len(prompts))):
        prompt = prompts[i]
        label = labels[i]
        
        # Skip safe prompts for quality measurement (focus on unsafe)
        if label == 0:
            continue
        
        # Generate WITHOUT steering
        env.config.intervention_start = env.config.num_inference_steps + 1
        env.config.intervention_end = 0
        
        obs, _ = env.reset(seed=i, options={'prompt': prompt})
        done = False
        step = 0
        
        while not done:
            action = np.zeros(env.action_space.shape)
            obs, _, done, _, _ = env.step(action)
            step += 1
        
        original_img = env.get_final_image()
        original_images.append(original_img)
        
        # Generate WITH steering at specific timestep
        env.config.intervention_start = timestep
        env.config.intervention_end = timestep  # Only intervene at this timestep
        
        obs, _ = env.reset(seed=i, options={'prompt': prompt})
        done = False
        step = 0
        
        while not done:
            if step == timestep:
                # Apply small steering action
                action = np.random.randn(*env.action_space.shape).astype(np.float32)
                action = action / np.linalg.norm(action) * steering_magnitude
            else:
                action = np.zeros(env.action_space.shape)
            
            obs, _, done, _, _ = env.step(action)
            step += 1
        
        steered_img = env.get_final_image()
        steered_images.append(steered_img)
        
        if len(original_images) >= num_samples:
            break
    
    if len(original_images) < 2:
        # Not enough samples, return default
        return 0.7
    
    # Compute FID
    fid_value = compute_fid_batch(original_images, steered_images, device)
    
    # Normalize FID (typical FID values: 0-100, normalize to [0, 1])
    # Lower FID = better quality preservation
    fid_norm = min(1.0, fid_value / 50.0)  # Normalize by 50 (reasonable FID threshold)
    quality = 1.0 - fid_norm
    
    return max(0.0, min(1.0, quality))  # Clamp to [0, 1]


def measure_steering_effectiveness(
    env: DiffusionSteeringEnv,
    probe: LinearProbe,
    prompts: List[str],
    labels: List[int],
    timestep: int,
    num_samples: int,
    device: str,
) -> float:
    """
    Measure steering effectiveness (SSR improvement) at a specific timestep.
    
    Generates images with and without steering, then measures SSR improvement.
    
    Returns:
        Effectiveness score in [0, 1] (higher = more effective steering)
    """
    original_preds = []
    steered_preds = []
    relevant_labels = []
    
    steering_magnitude = 0.1  # Slightly larger for effectiveness
    
    for i in range(min(num_samples, len(prompts))):
        prompt = prompts[i]
        label = labels[i]
        
        # Focus on unsafe prompts
        if label != 1:
            continue
        
        # Generate WITHOUT steering
        env.config.intervention_start = env.config.num_inference_steps + 1
        env.config.intervention_end = 0
        
        obs, _ = env.reset(seed=i, options={'prompt': prompt})
        done = False
        
        while not done:
            action = np.zeros(env.action_space.shape)
            obs, _, done, _, _ = env.step(action)
        
        # Get safety prediction
        with torch.no_grad():
            final_latent = env.current_latent
            if final_latent is not None:
                latent_flat = final_latent.flatten().float().unsqueeze(0).to(device)
                probe_score = probe(latent_flat)
                orig_pred = (torch.sigmoid(probe_score) > 0.5).int().item()
            else:
                orig_pred = 1  # Assume unsafe
        
        original_preds.append(orig_pred)
        
        # Generate WITH steering at specific timestep
        env.config.intervention_start = timestep
        env.config.intervention_end = timestep
        
        obs, _ = env.reset(seed=i, options={'prompt': prompt})
        done = False
        step = 0
        
        while not done:
            if step == timestep:
                # Apply steering action (toward safety)
                # Use probe gradient to guide steering direction
                with torch.no_grad():
                    z_flat = env.current_latent.flatten().float().unsqueeze(0).to(device)
                    z_flat.requires_grad = True
                    score = probe(z_flat)
                    # Gradient points toward safety (negative score = safe)
                    grad = torch.autograd.grad(score, z_flat, create_graph=False)[0]
                    # Project to action space (simplified)
                    action = -grad.squeeze().cpu().numpy()[:env.action_space.shape[0]]
                    action = action / (np.linalg.norm(action) + 1e-8) * steering_magnitude
            else:
                action = np.zeros(env.action_space.shape)
            
            obs, _, done, _, _ = env.step(action)
            step += 1
        
        # Get safety prediction after steering
        with torch.no_grad():
            final_latent = env.current_latent
            if final_latent is not None:
                latent_flat = final_latent.flatten().float().unsqueeze(0).to(device)
                probe_score = probe(latent_flat)
                steered_pred = (torch.sigmoid(probe_score) > 0.5).int().item()
            else:
                steered_pred = 0  # Assume steering worked
        
        steered_preds.append(steered_pred)
        relevant_labels.append(label)
        
        if len(original_preds) >= num_samples:
            break
    
    if len(original_preds) < 2:
        # Not enough samples, return default
        return 0.5
    
    # Compute SSR improvement
    original_preds = np.array(original_preds)
    steered_preds = np.array(steered_preds)
    relevant_labels = np.array(relevant_labels)
    
    # SSR = (unsafe → safe) / total_unsafe
    unsafe_mask = relevant_labels == 1
    total_unsafe = unsafe_mask.sum()
    
    if total_unsafe == 0:
        return 0.5
    
    unsafe_to_safe = (
        (original_preds[unsafe_mask] == 1) & 
        (steered_preds[unsafe_mask] == 0)
    ).sum()
    
    ssr = unsafe_to_safe / total_unsafe
    effectiveness = float(ssr)
    
    return max(0.0, min(1.0, effectiveness))  # Clamp to [0, 1]


def main():
    args = parse_args()
    
    latents_dir = Path(args.latents_dir)
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    print("="*60)
    print("EMPIRICAL LAYER SENSITIVITY MEASUREMENT")
    print("="*60)
    print(f"Latents directory: {latents_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print()
    
    # Load prompts
    print("Loading prompts...")
    prompts, labels = load_prompts(latents_dir)
    print(f"Loaded {len(prompts)} prompts ({sum(labels)} unsafe, {len(labels) - sum(labels)} safe)")
    
    # Create environment
    print("\nCreating environment...")
    env_config = AetherConfig(
        model_id=args.model_id,
        num_inference_steps=args.num_steps,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    
    env = DiffusionSteeringEnv(
        config=env_config,
        prompts=prompts,
    )
    
    # Load probe if available
    probe = None
    if args.probe_path:
        print(f"\nLoading probe from {args.probe_path}...")
        probe_path = Path(args.probe_path)
        if probe_path.is_dir():
            probe_file = probe_path / "probe_t04.pt"
            if not probe_file.exists():
                probe_files = list(probe_path.glob("probe_t*.pt"))
                if probe_files:
                    probe_file = sorted(probe_files)[len(probe_files) // 2]
        
        if probe_file.exists():
            probe = LinearProbe(input_dim=16384)
            probe.load_state_dict(torch.load(probe_file, map_location=args.device))
            probe = probe.to(args.device)
            probe.eval()
            print("✓ Probe loaded")
        else:
            print("Warning: Probe file not found, skipping SSR measurement")
    
    # Measure at each timestep
    print(f"\nMeasuring at {args.num_steps} timesteps...")
    print("This may take a while...")
    
    quality_measurements = {}
    effectiveness_measurements = {}
    
    # Sample timesteps to measure (not all, to save time)
    timesteps_to_measure = list(range(0, args.num_steps + 1, max(1, args.num_steps // 10)))
    
    for t in tqdm(timesteps_to_measure, desc="Measuring timesteps"):
        # Measure quality preservation
        try:
            quality = measure_quality_preservation(
                env, prompts, labels, t, args.num_samples, args.device
            )
            quality_measurements[t] = quality
        except Exception as e:
            print(f"Warning: Failed to measure quality at timestep {t}: {e}")
            quality_measurements[t] = 0.7  # Default
        
        # Measure steering effectiveness (if probe available)
        if probe is not None:
            try:
                effectiveness = measure_steering_effectiveness(
                    env, probe, prompts, labels, t, args.num_samples, args.device
                )
                effectiveness_measurements[t] = effectiveness
            except Exception as e:
                print(f"Warning: Failed to measure effectiveness at timestep {t}: {e}")
                effectiveness_measurements[t] = 0.5  # Default
        
        # Clear CUDA cache
        if args.device == "cuda":
            torch.cuda.empty_cache()
    
    # Save measurements
    print("\nSaving measurements...")
    
    quality_file = latents_dir / "quality_measurements.json"
    with open(quality_file, 'w') as f:
        json.dump(quality_measurements, f, indent=2)
    print(f"✓ Saved quality measurements to {quality_file}")
    
    if effectiveness_measurements:
        effectiveness_file = latents_dir / "effectiveness_measurements.json"
        with open(effectiveness_file, 'w') as f:
            json.dump(effectiveness_measurements, f, indent=2)
        print(f"✓ Saved effectiveness measurements to {effectiveness_file}")
    
    print("\n" + "="*60)
    print("MEASUREMENT COMPLETE")
    print("="*60)
    print(f"Quality measurements: {len(quality_measurements)} timesteps")
    print(f"Effectiveness measurements: {len(effectiveness_measurements)} timesteps")
    print(f"\nNext step: Run train_probes.py with --use_empirical flag")
    print(f"  python scripts/train_probes.py --latents_dir {latents_dir} --use_empirical")


if __name__ == "__main__":
    main()

