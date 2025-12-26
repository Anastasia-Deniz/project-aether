"""
Project Aether - Generate Images from Collected Latents
Simple script to decode latents back to images for visualization.

Usage:
    python scripts/generate_images_from_latents.py \
        --latents_dir ./data/latents/run_XXXXX \
        --timestep 8 \
        --num_samples 20
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from collected latents")
    
    parser.add_argument(
        "--latents_dir",
        type=str,
        required=True,
        help="Directory containing collected latents"
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=8,
        help="Timestep to decode (use final timestep for final images)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Model ID for decoding (should match collection model)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (None = all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: latents_dir/images_t{timestep})"
    )
    
    return parser.parse_args()


def decode_latent_to_image(pipe, latent: np.ndarray, device: str) -> np.ndarray:
    """Decode a latent vector to an image."""
    # Reshape latent from flat to (1, 4, 64, 64)
    latent_4d = latent.reshape(1, 4, 64, 64)
    latent_tensor = torch.from_numpy(latent_4d).float().to(device)
    
    # Convert to float16 if model uses it
    if pipe.vae.dtype == torch.float16:
        latent_tensor = latent_tensor.half()
    
    # Decode
    with torch.no_grad():
        image = pipe.vae.decode(
            latent_tensor / pipe.vae.config.scaling_factor,
            return_dict=False
        )[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    
    return image


def main():
    args = parse_args()
    
    latents_dir = Path(args.latents_dir)
    
    # Load latents
    latents_file = latents_dir / "latents" / f"timestep_{args.timestep:02d}.npz"
    if not latents_file.exists():
        latents_file = latents_dir / f"timestep_{args.timestep:02d}.npz"
    
    if not latents_file.exists():
        raise FileNotFoundError(f"Latents file not found: {latents_file}")
    
    print(f"Loading latents from: {latents_file}")
    data = np.load(latents_file)
    X = data['X']
    y = data['y']
    
    print(f"Loaded {len(X)} samples")
    print(f"Label distribution: Safe={np.sum(y==0)}, Unsafe={np.sum(y==1)}")
    
    # Load prompts
    safe_prompts = []
    unsafe_prompts = []
    
    safe_file = latents_dir / "safe_prompts.json"
    unsafe_file = latents_dir / "unsafe_prompts.json"
    
    if safe_file.exists():
        with open(safe_file, 'r', encoding='utf-8') as f:
            safe_prompts = json.load(f)
    
    if unsafe_file.exists():
        with open(unsafe_file, 'r', encoding='utf-8') as f:
            unsafe_prompts = json.load(f)
    
    all_prompts = [p['prompt'] for p in safe_prompts] + [p['prompt'] for p in unsafe_prompts]
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = latents_dir / f"images_t{args.timestep:02d}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    safe_dir = output_dir / "safe"
    unsafe_dir = output_dir / "unsafe"
    safe_dir.mkdir(exist_ok=True)
    unsafe_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load model
    print(f"\nLoading model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(args.device)
    pipe.vae.eval()
    print("Model loaded")
    
    # Determine samples to process
    num_samples = args.num_samples if args.num_samples else len(X)
    indices = np.arange(min(num_samples, len(X)))
    
    print(f"\nGenerating {len(indices)} images...")
    
    # Generate images
    for i in tqdm(indices, desc="Decoding"):
        label = y[i]
        prompt = all_prompts[i] if i < len(all_prompts) else f"Sample {i}"
        
        try:
            # Decode image
            image = decode_latent_to_image(pipe, X[i], args.device)
            
            # Save image
            if label == 0:  # Safe
                img_path = safe_dir / f"safe_{i:04d}.png"
            else:  # Unsafe
                img_path = unsafe_dir / f"unsafe_{i:04d}.png"
            
            img = Image.fromarray(image)
            img.save(img_path)
            
            # Save prompt as text file
            prompt_path = img_path.with_suffix('.txt')
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
        
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    # Create HTML viewer
    html_file = output_dir / "viewer.html"
    create_html_viewer(html_file, safe_dir, unsafe_dir, all_prompts, y[:len(indices)])
    
    print(f"\n✓ Complete!")
    print(f"  Images saved to: {output_dir}")
    print(f"  Safe images: {safe_dir}")
    print(f"  Unsafe images: {unsafe_dir}")
    print(f"  Viewer: {html_file}")


def create_html_viewer(html_file: Path, safe_dir: Path, unsafe_dir: Path, prompts: List[str], labels: np.ndarray):
    """Create an HTML file to view all images."""
    safe_images = sorted(safe_dir.glob("*.png"))
    unsafe_images = sorted(unsafe_dir.glob("*.png"))
    
    # Escape curly braces in CSS by doubling them
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Generated Images - Safe vs Unsafe</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .image-item {{
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            background-color: #fafafa;
        }}
        .image-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .image-item .prompt {{
            margin-top: 8px;
            font-size: 11px;
            color: #666;
            word-wrap: break-word;
        }}
        .safe-section h2 {{
            border-bottom-color: #27ae60;
        }}
        .unsafe-section h2 {{
            border-bottom-color: #e74c3c;
        }}
    </style>
</head>
<body>
    <h1>Generated Images - Safe vs Unsafe</h1>
    <p>Total: <strong>{len(safe_images)} safe</strong> + <strong>{len(unsafe_images)} unsafe</strong> images</p>
    
    <div class="section safe-section">
        <h2>Safe Images ({len(safe_images)})</h2>
        <div class="image-grid">
"""
    
    # Add safe images
    for img_path in safe_images:
        rel_path = img_path.relative_to(html_file.parent)
        prompt_path = img_path.with_suffix('.txt')
        prompt = "No prompt available"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        
        html_content += f"""
            <div class="image-item">
                <img src="{rel_path}" alt="Safe image">
                <div class="prompt">{prompt[:100]}{'...' if len(prompt) > 100 else ''}</div>
            </div>
"""
    
    html_content += f"""
        </div>
    </div>
    
    <div class="section unsafe-section">
        <h2>Unsafe Images ({len(unsafe_images)})</h2>
        <div class="image-grid">
"""
    
    # Add unsafe images
    for img_path in unsafe_images:
        rel_path = img_path.relative_to(html_file.parent)
        prompt_path = img_path.with_suffix('.txt')
        prompt = "No prompt available"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        
        html_content += f"""
            <div class="image-item">
                <img src="{rel_path}" alt="Unsafe image">
                <div class="prompt">{prompt[:100]}{'...' if len(prompt) > 100 else ''}</div>
            </div>
"""
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    main()

