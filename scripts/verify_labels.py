"""
Project Aether - Label Verification Script
Verifies that generated images match their prompt-based labels using a safety classifier.

This script:
1. Loads generated images from latents
2. Uses a safety classifier (CLIP-based or probe-based) to verify labels
3. Filters out mismatched samples
4. Creates a cleaned dataset with verified labels

Usage:
    python scripts/verify_labels.py \
        --latents_dir ./data/latents/run_YYYYMMDD_HHMMSS \
        --output_dir ./data/latents/run_YYYYMMDD_HHMMSS_verified \
        --device cuda
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionPipeline

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.linear_probe import LinearProbe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify labels using safety classifier"
    )
    
    parser.add_argument(
        "--latents_dir",
        type=str,
        required=True,
        help="Directory containing collected latents"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for verified latents (default: latents_dir + '_verified')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Model ID for decoding images"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["clip", "probe", "both"],
        default="clip",
        help="Verification method: 'clip' (CLIP-based), 'probe' (probe-based), 'both'"
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        default=None,
        help="Path to trained probe (required if method='probe' or 'both')"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for label verification (0-1)"
    )
    parser.add_argument(
        "--min_match_pct",
        type=float,
        default=0.8,
        help="Minimum percentage of samples that must match labels (otherwise warns)"
    )
    
    return parser.parse_args()


class CLIPSafetyClassifier:
    """Simple CLIP-based safety classifier using text-image similarity."""
    
    def __init__(self, device: str = "cuda"):
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = device
        print("Loading CLIP model for safety classification...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        
        # Define safe/unsafe text prompts
        self.safe_texts = [
            "a safe, appropriate, family-friendly image",
            "a clean, professional photograph",
            "a wholesome, suitable image",
        ]
        self.unsafe_texts = [
            "nudity, sexual content, inappropriate",
            "explicit, adult content, NSFW",
            "inappropriate, unsafe, adult material",
        ]
        
        # Encode text prompts
        with torch.no_grad():
            safe_inputs = self.processor(text=self.safe_texts, return_tensors="pt", padding=True)
            unsafe_inputs = self.processor(text=self.unsafe_texts, return_tensors="pt", padding=True)
            
            self.safe_embeds = self.model.get_text_features(**{k: v.to(device) for k, v in safe_inputs.items()})
            self.unsafe_embeds = self.model.get_text_features(**{k: v.to(device) for k, v in unsafe_inputs.items()})
    
    @torch.no_grad()
    def predict(self, images: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict safety for images.
        
        Args:
            images: Array of images (N, H, W, C) in uint8 format
            debug: If True, return additional debug info
            
        Returns:
            (predictions, probabilities) where predictions: 0=safe, 1=unsafe
            If debug=True, also returns (safe_sim, unsafe_sim) tuple
        """
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        
        # Process images
        inputs = self.processor(images=[Image.fromarray(img) for img in images], return_tensors="pt", padding=True)
        image_features = self.model.get_image_features(**{k: v.to(self.device) for k, v in inputs.items()})
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        safe_embeds = self.safe_embeds / self.safe_embeds.norm(dim=-1, keepdim=True)
        unsafe_embeds = self.unsafe_embeds / self.unsafe_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        safe_sim = (image_features @ safe_embeds.T).mean(dim=1)  # Average over safe texts
        unsafe_sim = (image_features @ unsafe_embeds.T).mean(dim=1)  # Average over unsafe texts
        
        # Probability of being unsafe
        probs = torch.sigmoid(unsafe_sim - safe_sim).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        if debug:
            return preds, probs, (safe_sim.cpu().numpy(), unsafe_sim.cpu().numpy())
        
        return preds, probs


def decode_latent_to_image(pipe, latent: np.ndarray, device: str) -> np.ndarray:
    """Decode latent to image."""
    # Convert to tensor - handle float16 numpy arrays properly
    if latent.dtype == np.float16:
        # Convert float16 to float32 first, then to tensor
        latent = latent.astype(np.float32)
    
    latent_tensor = torch.from_numpy(latent).reshape(1, 4, 64, 64).to(device)
    
    # Match the VAE's dtype to avoid dtype mismatch errors
    vae_dtype = next(pipe.vae.parameters()).dtype
    latent_tensor = latent_tensor.to(dtype=vae_dtype)
    
    with torch.no_grad():
        image = pipe.vae.decode(
            latent_tensor / pipe.vae.config.scaling_factor,
            return_dict=False
        )[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    
    return image


def verify_labels(
    latents_dir: Path,
    output_dir: Path,
    device: str,
    model_id: str,
    method: str,
    probe_path: str = None,
    threshold: float = 0.7,
) -> Dict:
    """Verify labels and create cleaned dataset."""
    
    # Load latents
    print("Loading latents...")
    latents_file = latents_dir / "latents" / "timestep_21.npz"  # Final timestep
    if not latents_file.exists():
        # Try alternative path
        latents_file = latents_dir / "timestep_21.npz"
    
    if not latents_file.exists():
        # Find any timestep file
        timestep_files = sorted(latents_dir.glob("latents/timestep_*.npz"))
        if not timestep_files:
            timestep_files = sorted(latents_dir.glob("timestep_*.npz"))
        if timestep_files:
            latents_file = timestep_files[-1]  # Use last timestep
        else:
            raise FileNotFoundError(f"No latents found in {latents_dir}")
    
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
        with open(safe_file, 'r') as f:
            safe_prompts = json.load(f)
    if unsafe_file.exists():
        with open(unsafe_file, 'r') as f:
            unsafe_prompts = json.load(f)
    
    all_prompts = [p['prompt'] if isinstance(p, dict) else p for p in safe_prompts] + \
                  [p['prompt'] if isinstance(p, dict) else p for p in unsafe_prompts]
    
    # Load model for decoding
    print(f"Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.vae.eval()
    
    # Initialize classifiers
    classifiers = {}
    
    if method in ["clip", "both"]:
        print("Initializing CLIP classifier...")
        classifiers['clip'] = CLIPSafetyClassifier(device=device)
    
    if method in ["probe", "both"]:
        if probe_path is None:
            raise ValueError("probe_path required when method='probe' or 'both'")
        
        print(f"Loading probe from {probe_path}...")
        latent_dim = X.shape[1]
        
        # Handle probe path - could be directory or specific file
        probe_file = Path(probe_path)
        if probe_file.is_dir():
            # Try to find best timestep probe (use timestep 21/final)
            probe_file = probe_file / "probe_t21.pt"
            if not probe_file.exists():
                # Try timestep 20
                probe_file = probe_file.parent / "probe_t20.pt"
            if not probe_file.exists():
                # Use any probe file
                probe_files = list(Path(probe_path).glob("probe_t*.pt"))
                if probe_files:
                    probe_file = probe_files[-1]  # Use last timestep
                else:
                    raise FileNotFoundError(f"No probe files found in {probe_path}")
        
        probe = LinearProbe(input_dim=latent_dim)
        probe.load_state_dict(torch.load(probe_file, map_location=device))
        probe = probe.to(device)
        probe.eval()
        classifiers['probe'] = probe
    
    # Decode images and verify labels
    print("\nDecoding images and verifying labels...")
    verified_indices = []
    mismatches = []
    
    # Statistics for debugging
    stats = {
        'total': 0,
        'matches': 0,
        'mismatches': 0,
        'low_confidence': 0,
        'safe_correct': 0,
        'safe_incorrect': 0,
        'unsafe_correct': 0,
        'unsafe_incorrect': 0,
        'confidence_scores': [],
        'prob_scores': [],
    }
    
    for i in tqdm(range(len(X)), desc="Verifying"):
        stats['total'] += 1
        
        # Decode image
        try:
            image = decode_latent_to_image(pipe, X[i], device)
        except Exception as e:
            print(f"\nWarning: Failed to decode sample {i}: {e}")
            continue
        
        # Get predictions from classifiers
        predictions = {}
        
        if 'clip' in classifiers:
            preds, probs, (safe_sim, unsafe_sim) = classifiers['clip'].predict(image, debug=True)
            predictions['clip'] = (preds[0], probs[0])
            # Store debug info for first few samples
            if i < 5:
                print(f"\n  Sample {i} (label={y[i]}): safe_sim={safe_sim[0]:.3f}, unsafe_sim={unsafe_sim[0]:.3f}, "
                      f"prob={probs[0]:.3f}, pred={preds[0]}")
        
        if 'probe' in classifiers:
            latent_tensor = torch.from_numpy(X[i]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logit = classifiers['probe'](latent_tensor)
                prob = torch.sigmoid(-logit).item()  # Negative because probe predicts unsafe
                pred = 1 if prob < 0.5 else 0  # 0=safe, 1=unsafe
            predictions['probe'] = (pred, 1 - prob if pred == 0 else prob)
        
        # Combine predictions
        if method == "both":
            # Average probabilities, use majority vote
            avg_prob = np.mean([p[1] for p in predictions.values()])
            pred = 1 if avg_prob > 0.5 else 0
            confidence = abs(avg_prob - 0.5) * 2  # Normalize to [0, 1]
            prob_used = avg_prob
        else:
            pred, prob_used = predictions[method]
            confidence = abs(prob_used - 0.5) * 2
        
        # Store statistics
        stats['prob_scores'].append(prob_used)
        stats['confidence_scores'].append(confidence)
        
        # Check if prediction matches label
        true_label = int(y[i])
        matches = (pred == true_label)
        
        # Update statistics
        if matches:
            stats['matches'] += 1
            if true_label == 0:
                stats['safe_correct'] += 1
            else:
                stats['unsafe_correct'] += 1
        else:
            stats['mismatches'] += 1
            if true_label == 0:
                stats['safe_incorrect'] += 1
            else:
                stats['unsafe_incorrect'] += 1
        
        if matches and confidence >= threshold:
            verified_indices.append(i)
        else:
            reason = []
            if not matches:
                reason.append(f"prediction_mismatch (pred={pred}, true={true_label})")
            if confidence < threshold:
                reason.append(f"low_confidence ({confidence:.3f} < {threshold})")
                stats['low_confidence'] += 1
            
            mismatches.append({
                'index': i,
                'true_label': true_label,
                'predicted': pred,
                'prob': prob_used,
                'confidence': confidence,
                'reason': '; '.join(reason),
                'prompt': all_prompts[i] if i < len(all_prompts) else f"Sample {i}",
            })
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("VERIFICATION STATISTICS")
    print("="*60)
    print(f"Total samples: {stats['total']}")
    print(f"Matches: {stats['matches']} ({100*stats['matches']/stats['total']:.1f}%)")
    print(f"Mismatches: {stats['mismatches']} ({100*stats['mismatches']/stats['total']:.1f}%)")
    print(f"\nLabel accuracy:")
    print(f"  Safe (label=0): {stats['safe_correct']} correct, {stats['safe_incorrect']} incorrect")
    print(f"  Unsafe (label=1): {stats['unsafe_correct']} correct, {stats['unsafe_incorrect']} incorrect")
    print(f"\nConfidence statistics:")
    if stats['confidence_scores']:
        conf_scores = np.array(stats['confidence_scores'])
        print(f"  Mean: {np.mean(conf_scores):.3f}")
        print(f"  Median: {np.median(conf_scores):.3f}")
        print(f"  Min: {np.min(conf_scores):.3f}")
        print(f"  Max: {np.max(conf_scores):.3f}")
        print(f"  Below threshold ({threshold}): {stats['low_confidence']} ({100*stats['low_confidence']/stats['total']:.1f}%)")
    print(f"\nProbability statistics:")
    if stats['prob_scores']:
        prob_scores = np.array(stats['prob_scores'])
        print(f"  Mean: {np.mean(prob_scores):.3f}")
        print(f"  Median: {np.median(prob_scores):.3f}")
        print(f"  Min: {np.min(prob_scores):.3f}")
        print(f"  Max: {np.max(prob_scores):.3f}")
        # Show distribution
        safe_probs = [p for i, p in enumerate(prob_scores) if y[i] == 0]
        unsafe_probs = [p for i, p in enumerate(prob_scores) if y[i] == 1]
        if safe_probs:
            print(f"  Safe samples (label=0) - Mean prob: {np.mean(safe_probs):.3f}")
        if unsafe_probs:
            print(f"  Unsafe samples (label=1) - Mean prob: {np.mean(unsafe_probs):.3f}")
    print("="*60)
    
    # Show sample mismatches for debugging
    if len(mismatches) > 0:
        print(f"\nSample mismatches (first 10):")
        for mm in mismatches[:10]:
            print(f"  Index {mm['index']}: True={mm['true_label']}, Pred={mm['predicted']}, "
                  f"Prob={mm['prob']:.3f}, Conf={mm['confidence']:.3f}, Reason={mm['reason']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    
    print(f"\n✓ Verification complete!")
    print(f"  Verified samples: {len(verified_indices)}/{len(X)} ({100*len(verified_indices)/len(X):.1f}%)")
    print(f"  Mismatches: {len(mismatches)}")
    
    if len(verified_indices) < len(X) * 0.5:
        print(f"\n⚠ Warning: Less than 50% of samples verified! Check your data.")
    
    # Save verified dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    verified_latents_dir = output_dir / "latents"
    verified_latents_dir.mkdir(exist_ok=True)
    
    # Load all timesteps and filter
    print("\nFiltering all timesteps...")
    timestep_files = sorted(latents_dir.glob("latents/timestep_*.npz"))
    if not timestep_files:
        timestep_files = sorted(latents_dir.glob("timestep_*.npz"))
    
    verified_y = y[verified_indices]
    
    for timestep_file in timestep_files:
        t = int(timestep_file.stem.split("_")[1])
        data_t = np.load(timestep_file)
        X_t = data_t['X'][verified_indices]
        
        np.savez_compressed(
            verified_latents_dir / f"timestep_{t:02d}.npz",
            X=X_t,
            y=verified_y,
        )
    
    # Save verified prompts
    verified_safe = [safe_prompts[i] for i in verified_indices if y[i] == 0]
    verified_unsafe = [unsafe_prompts[i - len(safe_prompts)] for i in verified_indices if y[i] == 1 and i >= len(safe_prompts)]
    
    with open(output_dir / "safe_prompts.json", 'w') as f:
        json.dump(verified_safe, f, indent=2)
    with open(output_dir / "unsafe_prompts.json", 'w') as f:
        json.dump(verified_unsafe, f, indent=2)
    
    # Save mismatch report
    with open(output_dir / "mismatch_report.json", 'w') as f:
        json.dump(mismatches, f, indent=2)
    
    # Copy metadata
    metadata_file = latents_dir / "metadata.json"
    if metadata_file.exists():
        import shutil
        shutil.copy(metadata_file, output_dir / "metadata.json")
    
    # Update metadata
    metadata = {
        "original_dir": str(latents_dir),
        "num_original": len(X),
        "num_verified": len(verified_indices),
        "num_mismatches": len(mismatches),
        "verification_method": method,
        "threshold": threshold,
    }
    
    with open(output_dir / "verification_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Verified dataset saved to: {output_dir}")
    print(f"  Latents: {verified_latents_dir}")
    print(f"  Mismatch report: {output_dir / 'mismatch_report.json'}")
    
    return {
        'verified_indices': verified_indices,
        'mismatches': mismatches,
        'output_dir': output_dir,
    }


def main():
    args = parse_args()
    
    latents_dir = Path(args.latents_dir)
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = latents_dir.parent / f"{latents_dir.name}_verified"
    
    print("="*60)
    print("LABEL VERIFICATION")
    print("="*60)
    print(f"Input: {latents_dir}")
    print(f"Output: {output_dir}")
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold}")
    print("="*60)
    
    result = verify_labels(
        latents_dir=latents_dir,
        output_dir=output_dir,
        device=args.device,
        model_id=args.model_id,
        method=args.method,
        probe_path=args.probe_path,
        threshold=args.threshold,
    )
    
    match_pct = len(result['verified_indices']) / (len(result['verified_indices']) + len(result['mismatches']))
    if match_pct < args.min_match_pct:
        print(f"\n⚠ Warning: Only {match_pct*100:.1f}% of samples matched labels (expected ≥{args.min_match_pct*100:.1f}%)")
        print("  Consider:")
        print("  - Using stricter prompt filtering")
        print("  - Using a more explicit/uncensored model")
        print("  - Manually reviewing mismatch_report.json")
    else:
        print(f"\n✓ {match_pct*100:.1f}% of samples matched labels (≥{args.min_match_pct*100:.1f}% threshold)")


if __name__ == "__main__":
    main()

