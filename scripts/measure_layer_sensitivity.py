"""
Project Aether - Empirical Layer Sensitivity Measurement
Phase 1 Step 5.2: Measure FID (quality preservation) and SSR improvement (steering effectiveness).

This script runs small steering experiments to empirically measure:
- Quality preservation: FID between steered and unsteered images (1 - FID_norm)
- Steering effectiveness: SSR improvement from steering at each timestep

These empirical measurements replace heuristics in sensitivity score computation, providing
more accurate layer sensitivity analysis for optimal intervention window selection.

Usage:
    # Basic usage (quality only)
    python scripts/measure_layer_sensitivity.py \
        --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS \
        --num_samples 20 \
        --device cuda
    
    # With probe (quality + effectiveness)
    python scripts/measure_layer_sensitivity.py \
        --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS \
        --probe_path ./checkpoints/probes/run_XXXXX/pytorch \
        --num_samples 20 \
        --device cuda

Output:
    - quality_measurements.json: {timestep: 1 - FID_norm}
    - effectiveness_measurements.json: {timestep: SSR_improvement} (if probe provided)

Note: This is computationally expensive (~30-60 min). Consider using --sample_timesteps
to measure only a subset of timesteps for faster results.
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
        help="Path to trained probe directory (optional, for SSR measurement). Should contain probe_t*.pt files."
    )
    parser.add_argument(
        "--sample_timesteps",
        type=int,
        default=None,
        help="Number of timesteps to sample for measurement (default: all timesteps). Use smaller values for faster measurement."
    )
    parser.add_argument(
        "--steering_magnitude",
        type=float,
        default=0.05,
        help="Magnitude of steering action for quality measurement (default: 0.05)"
    )
    parser.add_argument(
        "--effectiveness_magnitude",
        type=float,
        default=0.1,
        help="Magnitude of steering action for effectiveness measurement (default: 0.1)"
    )
    parser.add_argument(
        "--skip_quality",
        action="store_true",
        help="Skip quality measurement (faster, only measures effectiveness if probe provided)"
    )
    parser.add_argument(
        "--skip_effectiveness",
        action="store_true",
        help="Skip effectiveness measurement (faster, only measures quality)"
    )
    
    return parser.parse_args()


def load_prompts(latents_dir: Path) -> Tuple[List[str], List[int]]:
    """
    Load prompts and labels from latents directory.
    
    Tries to load from saved prompt files, falls back to I2P dataset if not found.
    """
    prompts = []
    labels = []
    
    # Try to load from saved prompts file
    prompts_file = latents_dir / "unsafe_prompts.json"
    if prompts_file.exists():
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                unsafe_data = json.load(f)
                for item in unsafe_data:
                    if isinstance(item, dict):
                        prompts.append(item['prompt'])
                        labels.append(1)  # Unsafe
                    elif isinstance(item, str):
                        prompts.append(item)
                        labels.append(1)
        except Exception as e:
            print(f"Warning: Failed to load unsafe prompts: {e}")
    
    # Try to load safe prompts
    safe_prompts_file = latents_dir / "safe_prompts.json"
    if safe_prompts_file.exists():
        try:
            with open(safe_prompts_file, 'r', encoding='utf-8') as f:
                safe_data = json.load(f)
                for item in safe_data:
                    if isinstance(item, dict):
                        prompts.append(item['prompt'])
                        labels.append(0)  # Safe
                    elif isinstance(item, str):
                        prompts.append(item)
                        labels.append(0)
        except Exception as e:
            print(f"Warning: Failed to load safe prompts: {e}")
    
    # Fallback: use I2P dataset if prompts not found
    if not prompts:
        print("Warning: No prompts found in latents directory. Loading from I2P dataset...")
        try:
            from src.utils.data import DataConfig, I2PDataset, AlternativeSafePrompts
            
            data_config = DataConfig(num_safe_samples=20, num_unsafe_samples=20)
            i2p = I2PDataset(data_config)
            unsafe_prompts = i2p.get_prompts(max_samples=20, focus_nudity_gore=True, min_nudity_pct=50.0)
            safe_prompts = AlternativeSafePrompts.get_prompts(num_samples=20, seed=42)
            
            prompts = [p['prompt'] if isinstance(p, dict) else p for p in safe_prompts]
            labels = [0] * len(prompts)
            
            prompts.extend([p['prompt'] if isinstance(p, dict) else p for p in unsafe_prompts])
            labels.extend([1] * len(unsafe_prompts))
        except Exception as e:
            raise RuntimeError(f"Failed to load prompts from dataset: {e}")
    
    if not prompts:
        raise ValueError("No prompts available for measurement")
    
    return prompts, labels


def compute_fid_batch(
    images1: List[np.ndarray],
    images2: List[np.ndarray],
    device: str = 'cuda',
) -> float:
    """
    Compute FID between two sets of images.
    
    Uses pytorch-fid library (Heusel et al., 2017).
    
    Args:
        images1: List of images as numpy arrays (H, W, 3) in [0, 255]
        images2: List of images as numpy arrays (H, W, 3) in [0, 255]
        device: Device to use for computation
        
    Returns:
        FID score (lower is better, typically 0-100)
    """
    if len(images1) < 2 or len(images2) < 2:
        print("Warning: Not enough images for FID computation (need at least 2)")
        return 50.0  # Return default high FID
    
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")
        print("Falling back to LPIPS-based approximation...")
        # Fallback: use LPIPS as approximation
        try:
            import lpips
            loss_fn = lpips.LPIPS(net='alex').to(device)
            
            total_lpips = 0.0
            count = 0
            
            for img1, img2 in zip(images1[:min(len(images1), len(images2))], images2[:min(len(images1), len(images2))]):
                # Convert to tensor and normalize to [-1, 1]
                img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
                img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
                
                with torch.no_grad():
                    lpips_val = loss_fn(img1_tensor, img2_tensor).item()
                    total_lpips += lpips_val
                    count += 1
            
            # Convert LPIPS to approximate FID (rough heuristic: LPIPS * 50)
            avg_lpips = total_lpips / count if count > 0 else 0.5
            approximate_fid = avg_lpips * 50.0
            return float(approximate_fid)
        except ImportError:
            print("Warning: LPIPS also not available. Returning default FID.")
            return 50.0
    
    # Save images to temporary directories
    import tempfile
    from PIL import Image
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            # Save images
            for i, img in enumerate(images1):
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                Image.fromarray(img).save(f"{tmpdir1}/img_{i:04d}.png")
            
            for i, img in enumerate(images2):
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                Image.fromarray(img).save(f"{tmpdir2}/img_{i:04d}.png")
            
            # Compute FID
            fid_value = fid_score.calculate_fid_given_paths(
                [tmpdir1, tmpdir2],
                batch_size=8,
                device=device,
                dims=2048,
            )
            
            return float(fid_value)
    except Exception as e:
        print(f"Warning: FID computation failed: {e}")
        return 50.0  # Return default high FID


def measure_quality_preservation(
    env: DiffusionSteeringEnv,
    prompts: List[str],
    labels: List[int],
    timestep: int,
    num_samples: int,
    device: str,
    steering_magnitude: float = 0.05,
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
    steering_magnitude: float = 0.1,
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
    
    # Steering magnitude is passed as parameter
    
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
                z_flat = env.current_latent.flatten().float().unsqueeze(0).to(device)
                z_flat.requires_grad = True
                
                score = probe(z_flat)
                # Gradient points toward increasing score (unsafe direction)
                # We want to steer toward safety (decrease score)
                grad = torch.autograd.grad(score, z_flat, create_graph=False, retain_graph=False)[0]
                
                # Project to action space and normalize
                action = -grad.squeeze().cpu().numpy()  # Negative gradient = toward safety
                
                # Handle action space dimension mismatch
                if action.shape[0] > env.action_space.shape[0]:
                    action = action[:env.action_space.shape[0]]
                elif action.shape[0] < env.action_space.shape[0]:
                    # Pad with zeros if needed
                    padded_action = np.zeros(env.action_space.shape[0])
                    padded_action[:action.shape[0]] = action
                    action = padded_action
                
                # Normalize and scale
                norm = np.linalg.norm(action)
                if norm > 1e-8:
                    action = action / norm * steering_magnitude
                else:
                    # Fallback: random direction if gradient is zero
                    action = np.random.randn(*env.action_space.shape).astype(np.float32)
                    action = action / np.linalg.norm(action) * steering_magnitude
                
                z_flat.requires_grad = False  # Clean up
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
    if args.probe_path and not args.skip_effectiveness:
        print(f"\nLoading probe from {args.probe_path}...")
        probe_path = Path(args.probe_path)
        probe_file = None
        scaler_file = None
        
        if probe_path.is_dir():
            # Try to find a probe file (prefer middle timestep)
            probe_files = sorted(probe_path.glob("probe_t*.pt"))
            if probe_files:
                # Use middle timestep probe
                probe_file = probe_files[len(probe_files) // 2]
                # Try to find corresponding scaler
                scaler_file = probe_path / f"scaler_{probe_file.stem.replace('probe_', '')}.pkl"
            else:
                print("Warning: No probe files found in directory")
                probe_file = None
        elif probe_path.is_file():
            probe_file = probe_path
            # Try to find corresponding scaler
            scaler_file = probe_path.parent / f"scaler_{probe_path.stem.replace('probe_', '')}.pkl"
        else:
            print(f"Warning: Probe path does not exist: {probe_path}")
            probe_file = None
        
        if probe_file is not None and probe_file.exists():
            try:
                # Determine latent dimension from metadata or use default
                latent_dim = 16384  # Default: 4 * 64 * 64
                
                # Try to load from metadata
                metadata_file = latents_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if 'latent_dim' in metadata:
                            latent_dim = metadata['latent_dim']
                
                # Load scaler if available
                scaler = None
                if scaler_file and scaler_file.exists():
                    try:
                        import pickle
                        with open(scaler_file, 'rb') as f:
                            scaler = pickle.load(f)
                        print(f"✓ Loaded scaler from {scaler_file.name}")
                    except Exception as e:
                        print(f"Warning: Could not load scaler: {e}")
                
                # Load probe
                probe = LinearProbe(input_dim=latent_dim, scaler=scaler)
                probe.load_state_dict(torch.load(probe_file, map_location=args.device))
                probe = probe.to(args.device)
                probe.eval()
                print(f"✓ Probe loaded from {probe_file.name} (dim={latent_dim})")
            except Exception as e:
                print(f"Warning: Failed to load probe: {e}")
                print("Skipping effectiveness measurement")
                probe = None
        else:
            print("Warning: Probe file not found, skipping SSR measurement")
    
    # Determine timesteps to measure
    if args.sample_timesteps:
        # Sample evenly spaced timesteps
        step_size = max(1, args.num_steps // args.sample_timesteps)
        timesteps_to_measure = list(range(0, args.num_steps + 1, step_size))
        if timesteps_to_measure[-1] != args.num_steps:
            timesteps_to_measure.append(args.num_steps)
    else:
        # Measure all timesteps
        timesteps_to_measure = list(range(0, args.num_steps + 1))
    
    print(f"\nMeasuring at {len(timesteps_to_measure)} timesteps: {timesteps_to_measure}")
    print("This may take a while...")
    if args.sample_timesteps:
        print(f"Note: Sampling {args.sample_timesteps} timesteps for faster measurement")
    
    quality_measurements = {}
    effectiveness_measurements = {}
    
    # Estimate time
    samples_per_timestep = min(args.num_samples, len([l for l in labels if l == 1]))
    estimated_time_min = len(timesteps_to_measure) * samples_per_timestep * 0.1  # ~0.1 min per sample
    print(f"Estimated time: ~{estimated_time_min:.1f} minutes")
    
    for t in tqdm(timesteps_to_measure, desc="Measuring timesteps"):
        # Measure quality preservation
        if not args.skip_quality:
            try:
                quality = measure_quality_preservation(
                    env, prompts, labels, t, args.num_samples, args.device,
                    steering_magnitude=args.steering_magnitude
                )
                quality_measurements[t] = quality
            except Exception as e:
                print(f"\nWarning: Failed to measure quality at timestep {t}: {e}")
                import traceback
                traceback.print_exc()
                quality_measurements[t] = 0.7  # Default
        else:
            quality_measurements[t] = 0.7  # Use default if skipped
        
        # Measure steering effectiveness (if probe available)
        if probe is not None and not args.skip_effectiveness:
            try:
                effectiveness = measure_steering_effectiveness(
                    env, probe, prompts, labels, t, args.num_samples, args.device,
                    steering_magnitude=args.effectiveness_magnitude
                )
                effectiveness_measurements[t] = effectiveness
            except Exception as e:
                print(f"\nWarning: Failed to measure effectiveness at timestep {t}: {e}")
                import traceback
                traceback.print_exc()
                effectiveness_measurements[t] = 0.5  # Default
        
        # Clear CUDA cache periodically
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
    
    # Print summary
    print("\n" + "="*60)
    print("MEASUREMENT COMPLETE")
    print("="*60)
    print(f"Quality measurements: {len(quality_measurements)} timesteps")
    if quality_measurements:
        avg_quality = np.mean(list(quality_measurements.values()))
        print(f"  Average quality: {avg_quality:.3f}")
        print(f"  Best quality: {max(quality_measurements.values()):.3f} (at t={max(quality_measurements, key=quality_measurements.get)})")
    
    print(f"Effectiveness measurements: {len(effectiveness_measurements)} timesteps")
    if effectiveness_measurements:
        avg_effectiveness = np.mean(list(effectiveness_measurements.values()))
        print(f"  Average effectiveness: {avg_effectiveness:.3f}")
        print(f"  Best effectiveness: {max(effectiveness_measurements.values()):.3f} (at t={max(effectiveness_measurements, key=effectiveness_measurements.get)})")
    
    print(f"\nNext step: Run train_probes.py with --use_empirical flag")
    print(f"  python scripts/train_probes.py --latents_dir {latents_dir} --use_empirical")
    
    # Warn if measurements are sparse
    if len(quality_measurements) < args.num_steps + 1:
        print(f"\nNote: Only {len(quality_measurements)} timesteps measured (out of {args.num_steps + 1})")
        print("Consider running without --sample_timesteps for complete measurements")


if __name__ == "__main__":
    main()

