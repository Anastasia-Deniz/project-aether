"""
Project Aether - Latent Collection Script
Phase 1: Collect latent representations at each timestep for probing.

Usage:
    python scripts/collect_latents.py --config configs/collect_latents.yaml
    
    # Or with command line args:
    python scripts/collect_latents.py \
        --model_id rupeshs/LCM-runwayml-stable-diffusion-v1-5 \
        --num_samples 100 \
        --num_steps 8 \
        --output_dir ./data/latents
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import (
    DataConfig, 
    I2PDataset, 
    AlternativeSafePrompts,
    save_prompts
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect latents for linear probing")
    
    # Model settings
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="CompVis/stable-diffusion-v1-4",  # SD 1.4 - less censored
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--num_steps", 
        type=int, 
        default=20,  # SD 1.4 uses 20-50 steps
        help="Number of diffusion steps"
    )
    
    # Data settings
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=50,  # Reduced from 100 for faster initial experiments
        help="Number of samples per class (safe/unsafe)"
    )
    parser.add_argument(
        "--categories", 
        type=str, 
        nargs="+",
        default=["violence", "sexual", "shocking"],
        help="I2P categories to use for unsafe prompts"
    )
    parser.add_argument(
        "--min_inappropriate_pct",
        type=float,
        default=30.0,
        help="Minimum inappropriate percentage for I2P prompts"
    )
    parser.add_argument(
        "--hard_only",
        action="store_true",
        help="Only use 'hard' prompts (>50% inappropriate rating)"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/latents",
        help="Output directory for latents"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images alongside latents"
    )
    
    # Compute settings
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (1 recommended for memory)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


class LatentCollector:
    """Collects latent representations at multiple timesteps."""
    
    def __init__(
        self,
        model_id: str,
        num_steps: int = 20,
        device: str = "cuda",
    ):
        from diffusers import StableDiffusionPipeline
        
        self.device = device
        self.num_steps = num_steps
        
        print(f"Loading model: {model_id}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        
        # Freeze model
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        
        print(f"Model loaded on {device}")
        
    @torch.no_grad()
    def collect_trajectory(
        self,
        prompt: str,
        seed: int = 42,
        guidance_scale: float = 7.5,
    ) -> dict:
        """
        Run diffusion and collect latents at each timestep.
        
        Returns:
            Dict with:
                - 'latents': Dict[int, np.ndarray] - timestep -> latent
                - 'final_image': np.ndarray - decoded image
                - 'prompt': str
        """
        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Encode prompt
        do_cfg = guidance_scale > 1.0
        prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
        )
        
        if isinstance(prompt_embeds, tuple):
            prompt_embeds, negative_prompt_embeds = prompt_embeds[:2]
        else:
            negative_prompt_embeds = None
        
        # Set scheduler
        self.pipe.scheduler.set_timesteps(self.num_steps, device=self.device)
        
        # Initialize from noise
        latent = torch.randn(
            1, 4, 64, 64,
            device=self.device,
            dtype=torch.float16,
            generator=generator,
        )
        latent = latent * self.pipe.scheduler.init_noise_sigma
        
        # Collect latents at each timestep
        latents = {0: latent.cpu().numpy().astype(np.float16)}
        
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # Prepare input
            if do_cfg and negative_prompt_embeds is not None:
                latent_input = torch.cat([latent] * 2)
                latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)
                prompt_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                latent_input = self.pipe.scheduler.scale_model_input(latent, t)
                prompt_input = prompt_embeds
            
            # UNet forward
            noise_pred = self.pipe.unet(
                latent_input, t, encoder_hidden_states=prompt_input, return_dict=False
            )[0]
            
            # Apply CFG
            if do_cfg and negative_prompt_embeds is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            
            # Scheduler step
            latent = self.pipe.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
            
            # Store latent
            latents[i + 1] = latent.cpu().numpy().astype(np.float16)
        
        # Decode final image
        image = self.pipe.vae.decode(
            latent / self.pipe.vae.config.scaling_factor, return_dict=False
        )[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        return {
            'latents': latents,
            'final_image': image,
            'prompt': prompt,
        }


def main():
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    
    # Load prompts
    print("\n" + "="*60)
    print("Loading Prompts")
    print("="*60)
    
    # Load unsafe prompts from I2P
    data_config = DataConfig(
        num_unsafe_samples=args.num_samples,
        num_safe_samples=args.num_samples,
        i2p_categories=args.categories,
        random_seed=args.seed,
    )
    
    i2p = I2PDataset(data_config)
    unsafe_prompts = i2p.get_prompts(
        categories=args.categories,
        max_samples=args.num_samples,
        min_inappropriate_pct=args.min_inappropriate_pct,
        hard_only=args.hard_only,
    )
    
    # Load safe prompts
    safe_prompts = AlternativeSafePrompts.get_prompts(
        num_samples=args.num_samples,
        seed=args.seed,
    )
    
    # Save prompts for reference
    save_prompts(safe_prompts, run_dir / "safe_prompts.json")
    save_prompts(unsafe_prompts, run_dir / "unsafe_prompts.json")
    
    print(f"\nCollecting {len(safe_prompts)} safe + {len(unsafe_prompts)} unsafe samples")
    
    # Initialize collector
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)
    
    collector = LatentCollector(
        model_id=args.model_id,
        num_steps=args.num_steps,
        device=args.device,
    )
    
    # Collect latents
    print("\n" + "="*60)
    print("Collecting Latents")
    print("="*60)
    
    # Storage: timestep -> list of latents
    # Use defaultdict to handle any number of timesteps from scheduler
    from collections import defaultdict
    all_latents = defaultdict(list)
    all_labels = []
    all_prompts = []
    
    # Images directory
    if args.save_images:
        images_dir = run_dir / "images"
        images_dir.mkdir(exist_ok=True)
    
    # Filter prompts by CLIP token limit before collection
    from src.utils.data import is_prompt_valid
    
    safe_prompts_filtered = [p for p in safe_prompts if is_prompt_valid(p["prompt"], max_tokens=77)]
    unsafe_prompts_filtered = [p for p in unsafe_prompts if is_prompt_valid(p["prompt"], max_tokens=77)]
    
    skipped_safe = len(safe_prompts) - len(safe_prompts_filtered)
    skipped_unsafe = len(unsafe_prompts) - len(unsafe_prompts_filtered)
    
    if skipped_safe > 0 or skipped_unsafe > 0:
        print(f"\nFiltered prompts: skipped {skipped_safe} safe + {skipped_unsafe} unsafe prompts exceeding 77 CLIP tokens")
        print(f"Using {len(safe_prompts_filtered)} safe + {len(unsafe_prompts_filtered)} unsafe prompts")
    
    # Collect safe samples (label=0)
    print("\nCollecting safe samples...")
    for i, prompt_data in enumerate(tqdm(safe_prompts_filtered, desc="Safe")):
        prompt = prompt_data["prompt"]
        result = collector.collect_trajectory(prompt, seed=args.seed + i)
        
        for t, latent in result['latents'].items():
            all_latents[t].append(latent.flatten())
        
        all_labels.append(0)
        all_prompts.append(prompt)
        
        if args.save_images:
            from PIL import Image
            img = Image.fromarray(result['final_image'])
            img.save(images_dir / f"safe_{i:04d}.png")
    
    # Collect unsafe samples (label=1)
    print("\nCollecting unsafe samples...")
    for i, prompt_data in enumerate(tqdm(unsafe_prompts_filtered, desc="Unsafe")):
        prompt = prompt_data["prompt"]
        result = collector.collect_trajectory(
            prompt, 
            seed=args.seed + len(safe_prompts_filtered) + i
        )
        
        for t, latent in result['latents'].items():
            all_latents[t].append(latent.flatten())
        
        all_labels.append(1)
        all_prompts.append(prompt)
        
        if args.save_images:
            from PIL import Image
            img = Image.fromarray(result['final_image'])
            img.save(images_dir / f"unsafe_{i:04d}.png")
    
    # Save latents
    print("\n" + "="*60)
    print("Saving Latents")
    print("="*60)
    
    labels = np.array(all_labels)
    
    # Save per-timestep
    latents_dir = run_dir / "latents"
    latents_dir.mkdir(exist_ok=True)
    
    # Get actual timesteps from collected data
    timesteps = sorted(all_latents.keys())
    print(f"  Collected {len(timesteps)} timesteps: {timesteps[0]} to {timesteps[-1]}")
    
    for t in timesteps:
        X = np.stack(all_latents[t])
        np.savez_compressed(
            latents_dir / f"timestep_{t:02d}.npz",
            X=X,
            y=labels,
        )
        print(f"  Saved timestep {t}: X.shape={X.shape}, labels={len(labels)}")
    
    # Save metadata
    metadata = {
        "model_id": args.model_id,
        "num_steps": args.num_steps,
        "num_timesteps_collected": len(timesteps),
        "timesteps": timesteps,
        "num_safe": len(safe_prompts),
        "num_unsafe": len(unsafe_prompts),
        "categories": args.categories,
        "seed": args.seed,
        "timestamp": timestamp,
        "latent_dim": all_latents[timesteps[0]][0].shape[0],
    }
    
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Collection complete!")
    print(f"  Output: {run_dir}")
    print(f"  Latents: {latents_dir}")
    print(f"  Total samples: {len(labels)}")
    print(f"  Label distribution: {np.bincount(labels)}")
    
    return run_dir


if __name__ == "__main__":
    main()

