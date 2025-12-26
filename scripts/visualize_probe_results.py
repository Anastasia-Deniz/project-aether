"""
Project Aether - Probe Visualization Script
Visualizes safe vs unsafe images and probe predictions to verify accuracy.

Usage:
    python scripts/visualize_probe_results.py \
        --latents_dir ./data/latents/run_XXXXX \
        --probe_dir ./checkpoints/probes/run_XXXXX/pytorch/ \
        --timestep 4
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.linear_probe import LinearProbe
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize probe results with images")
    
    parser.add_argument(
        "--latents_dir",
        type=str,
        required=True,
        help="Directory containing collected latents"
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        required=True,
        help="Directory containing trained probes (pytorch folder)"
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=4,
        help="Timestep to visualize (use best timestep from sensitivity analysis)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Model ID for decoding latents to images"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to visualize per class"
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
        default="./outputs/visualizations",
        help="Output directory for visualizations"
    )
    
    return parser.parse_args()


def load_latents_and_prompts(latents_dir: Path, timestep: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load latents, labels, and prompts for a specific timestep."""
    latents_dir = Path(latents_dir)
    
    # Load latents
    latents_file = latents_dir / "latents" / f"timestep_{timestep:02d}.npz"
    if not latents_file.exists():
        # Try without /latents subdirectory
        latents_file = latents_dir / f"timestep_{timestep:02d}.npz"
    
    if not latents_file.exists():
        raise FileNotFoundError(f"Latents file not found: {latents_file}")
    
    data = np.load(latents_file)
    X = data['X']
    y = data['y']
    
    # Load prompts
    safe_prompts_file = latents_dir / "safe_prompts.json"
    unsafe_prompts_file = latents_dir / "unsafe_prompts.json"
    
    prompts = []
    if safe_prompts_file.exists():
        with open(safe_prompts_file, 'r') as f:
            safe_prompts = json.load(f)
            prompts.extend([p['prompt'] for p in safe_prompts])
    
    if unsafe_prompts_file.exists():
        with open(unsafe_prompts_file, 'r') as f:
            unsafe_prompts = json.load(f)
            prompts.extend([p['prompt'] for p in unsafe_prompts])
    
    # Ensure prompts match labels
    if len(prompts) != len(y):
        print(f"Warning: {len(prompts)} prompts but {len(y)} labels. Using indices.")
        prompts = [f"Sample {i}" for i in range(len(y))]
    
    return X, y, prompts


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


def load_probe(probe_dir: Path, timestep: int, latent_dim: int, device: str):
    """Load a trained probe."""
    probe_file = probe_dir / f"probe_t{timestep:02d}.pt"
    
    if not probe_file.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_file}")
    
    probe = LinearProbe(input_dim=latent_dim)
    probe.load_state_dict(torch.load(probe_file, map_location=device))
    probe = probe.to(device)
    probe.eval()
    
    return probe


def get_probe_prediction(probe, latent: np.ndarray, device: str) -> Tuple[float, int]:
    """Get probe prediction for a latent."""
    latent_tensor = torch.from_numpy(latent).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logit = probe(latent_tensor)
        prob_unsafe = torch.sigmoid(logit).item()
        pred = 1 if prob_unsafe > 0.5 else 0
    
    return prob_unsafe, pred


def visualize_probe_results(
    latents_dir: Path,
    probe_dir: Path,
    timestep: int,
    model_id: str,
    num_samples: int,
    device: str,
    output_dir: Path,
):
    """Main visualization function."""
    print("="*60)
    print("PROBE VISUALIZATION")
    print("="*60)
    print(f"Latents: {latents_dir}")
    print(f"Probe: {probe_dir}")
    print(f"Timestep: {timestep}")
    print(f"Model: {model_id}")
    
    # Load data
    print("\n--- Loading Data ---")
    X, y, prompts = load_latents_and_prompts(latents_dir, timestep)
    latent_dim = X.shape[1]
    
    print(f"Loaded {len(X)} samples")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Latent dimension: {latent_dim}")
    
    # Load probe
    print("\n--- Loading Probe ---")
    probe_file = probe_dir / f"probe_t{timestep:02d}.pt"
    probe = load_probe(probe_dir, timestep, latent_dim, device)
    print(f"Probe loaded: {probe_file}")
    
    # Load model for decoding
    print("\n--- Loading Model for Decoding ---")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.vae.eval()
    print("Model loaded")
    
    # Get predictions
    print("\n--- Computing Probe Predictions ---")
    predictions = []
    probabilities = []
    
    for i, latent in enumerate(tqdm(X, desc="Predicting")):
        prob, pred = get_probe_prediction(probe, latent, device)
        predictions.append(pred)
        probabilities.append(prob)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Compute accuracy
    accuracy = (predictions == y).mean()
    print(f"\nProbe Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Safe samples (y=0): {(predictions[y==0] == 0).mean():.3f}")
    print(f"  Unsafe samples (y=1): {(predictions[y==1] == 1).mean():.3f}")
    
    # Find correct and incorrect predictions
    correct_mask = predictions == y
    incorrect_mask = predictions != y
    
    print(f"\nCorrect predictions: {correct_mask.sum()}/{len(y)}")
    print(f"Incorrect predictions: {incorrect_mask.sum()}/{len(y)}")
    
    # Decode images (sample from each category)
    print("\n--- Decoding Images ---")
    
    # Sample indices
    safe_indices = np.where(y == 0)[0]
    unsafe_indices = np.where(y == 1)[0]
    
    # Get correct and incorrect samples
    safe_correct = safe_indices[correct_mask[safe_indices]][:num_samples]
    safe_incorrect = safe_indices[incorrect_mask[safe_indices]][:num_samples]
    unsafe_correct = unsafe_indices[correct_mask[unsafe_indices]][:num_samples]
    unsafe_incorrect = unsafe_indices[incorrect_mask[unsafe_indices]][:num_samples]
    
    # Decode images
    images_data = {}
    
    def decode_samples(indices, label, status):
        images = []
        for idx in tqdm(indices, desc=f"Decoding {status} {label}"):
            try:
                img = decode_latent_to_image(pipe, X[idx], device)
                images.append({
                    'image': img,
                    'prompt': prompts[idx] if idx < len(prompts) else f"Sample {idx}",
                    'label': y[idx],
                    'pred': predictions[idx],
                    'prob': probabilities[idx],
                    'correct': correct_mask[idx],
                })
            except Exception as e:
                print(f"Error decoding sample {idx}: {e}")
        return images
    
    images_data['safe_correct'] = decode_samples(safe_correct, 'safe', 'correct')
    images_data['safe_incorrect'] = decode_samples(safe_incorrect, 'safe', 'incorrect')
    images_data['unsafe_correct'] = decode_samples(unsafe_correct, 'unsafe', 'correct')
    images_data['unsafe_incorrect'] = decode_samples(unsafe_incorrect, 'unsafe', 'incorrect')
    
    # Create visualization
    print("\n--- Creating Visualization ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(
        f'Probe Visualization - Timestep {timestep}\n'
        f'Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)',
        fontsize=16, fontweight='bold'
    )
    
    # Plot 1: Safe - Correct Predictions
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title('Safe Images - Correctly Classified (✓)', fontsize=12, fontweight='bold', color='green')
    plot_image_grid(ax1, images_data['safe_correct'][:10], 'Safe (Correct)')
    
    # Plot 2: Safe - Incorrect Predictions
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_title('Safe Images - Incorrectly Classified as Unsafe (✗)', fontsize=12, fontweight='bold', color='red')
    plot_image_grid(ax2, images_data['safe_incorrect'][:10], 'Safe (Incorrect)')
    
    # Plot 3: Unsafe - Correct Predictions
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title('Unsafe Images - Correctly Classified (✓)', fontsize=12, fontweight='bold', color='green')
    plot_image_grid(ax3, images_data['unsafe_correct'][:10], 'Unsafe (Correct)')
    
    # Plot 4: Unsafe - Incorrect Predictions
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_title('Unsafe Images - Incorrectly Classified as Safe (✗)', fontsize=12, fontweight='bold', color='red')
    plot_image_grid(ax4, images_data['unsafe_incorrect'][:10], 'Unsafe (Incorrect)')
    
    # Plot 5: Confusion Matrix
    ax5 = fig.add_subplot(gs[2, :2])
    plot_confusion_matrix(ax5, y, predictions)
    
    # Plot 6: Probability Distribution
    ax6 = fig.add_subplot(gs[2, 2:])
    plot_probability_distribution(ax6, y, probabilities)
    
    # Plot 7: Accuracy by Category
    ax7 = fig.add_subplot(gs[3, :])
    plot_accuracy_breakdown(ax7, y, predictions, prompts)
    
    # Save figure
    output_file = output_dir / f"probe_visualization_t{timestep:02d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    
    # Save detailed results
    results_file = output_dir / f"probe_results_t{timestep:02d}.json"
    results = {
        'timestep': timestep,
        'accuracy': float(accuracy),
        'total_samples': int(len(y)),
        'correct': int(correct_mask.sum()),
        'incorrect': int(incorrect_mask.sum()),
        'safe_accuracy': float((predictions[y==0] == 0).mean()),
        'unsafe_accuracy': float((predictions[y==1] == 1).mean()),
        'samples': []
    }
    
    # Add sample details
    for idx in range(min(50, len(y))):  # Save first 50 samples
        results['samples'].append({
            'index': int(idx),
            'prompt': prompts[idx] if idx < len(prompts) else f"Sample {idx}",
            'true_label': int(y[idx]),
            'predicted_label': int(predictions[idx]),
            'probability_unsafe': float(probabilities[idx]),
            'correct': bool(correct_mask[idx]),
        })
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Safe samples: {(predictions[y==0] == 0).mean():.3f}")
    print(f"  Unsafe samples: {(predictions[y==1] == 1).mean():.3f}")
    print(f"\nOutput: {output_dir}")
    
    return output_dir


def plot_image_grid(ax, images_data: List[Dict], title: str):
    """Plot a grid of images with labels."""
    if not images_data:
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    n_images = min(len(images_data), 10)
    cols = 5
    rows = (n_images + cols - 1) // cols
    
    ax.axis('off')
    
    for i, data in enumerate(images_data[:n_images]):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x0 = col / cols
        y0 = 1 - (row + 1) / rows
        width = 1 / cols
        height = 1 / rows
        
        # Create subplot for image
        img_ax = ax.inset_axes([x0, y0, width, height])
        img_ax.imshow(data['image'])
        img_ax.axis('off')
        
        # Add label
        prob = data['prob']
        pred = data['pred']
        label_text = f"P={prob:.2f}"
        color = 'green' if data['correct'] else 'red'
        img_ax.text(0.5, -0.1, label_text, transform=img_ax.transAxes,
                   ha='center', va='top', fontsize=8, color=color, fontweight='bold')


def plot_confusion_matrix(ax, y_true: np.ndarray, y_pred: np.ndarray):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Safe', 'Unsafe'])
    ax.set_yticklabels(['Safe', 'Unsafe'])


def plot_probability_distribution(ax, y_true: np.ndarray, probabilities: np.ndarray):
    """Plot probability distribution for safe vs unsafe."""
    safe_probs = probabilities[y_true == 0]
    unsafe_probs = probabilities[y_true == 1]
    
    ax.hist(safe_probs, bins=20, alpha=0.6, label='Safe (y=0)', color='blue', density=True)
    ax.hist(unsafe_probs, bins=20, alpha=0.6, label='Unsafe (y=1)', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    ax.set_xlabel('P(Unsafe)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Probability Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_accuracy_breakdown(ax, y_true: np.ndarray, y_pred: np.ndarray, prompts: List[str]):
    """Plot accuracy breakdown by category."""
    # Group by first few words of prompt (simple heuristic)
    categories = {}
    for i, prompt in enumerate(prompts):
        # Use first 3 words as category
        category = ' '.join(prompt.split()[:3])
        if category not in categories:
            categories[category] = {'correct': 0, 'total': 0}
        categories[category]['total'] += 1
        if y_pred[i] == y_true[i]:
            categories[category]['correct'] += 1
    
    # Sort by total count
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
    
    if not sorted_cats:
        ax.text(0.5, 0.5, 'No category data', ha='center', va='center')
        ax.axis('off')
        return
    
    cat_names = [cat[:30] for cat, _ in sorted_cats]
    accuracies = [data['correct'] / data['total'] for _, data in sorted_cats]
    counts = [data['total'] for _, data in sorted_cats]
    
    bars = ax.barh(range(len(cat_names)), accuracies, color='steelblue')
    ax.set_yticks(range(len(cat_names)))
    ax.set_yticklabels(cat_names, fontsize=8)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy by Prompt Category (Top 10)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (acc, count) in enumerate(zip(accuracies, counts)):
        ax.text(acc + 0.02, i, f'n={count}', va='center', fontsize=8)


if __name__ == "__main__":
    args = parse_args()
    
    visualize_probe_results(
        latents_dir=Path(args.latents_dir),
        probe_dir=Path(args.probe_dir),
        timestep=args.timestep,
        model_id=args.model_id,
        num_samples=args.num_samples,
        device=args.device,
        output_dir=Path(args.output_dir),
    )

