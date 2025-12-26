"""
Project Aether - Linear Probe Training Script
Phase 1: Train linear probes at each timestep to validate concept separability.

Usage:
    python scripts/train_probes.py --latents_dir ./data/latents/run_XXXXX
    
This script:
1. Loads collected latents
2. Trains a linear probe at each timestep
3. Computes accuracy and AUC metrics
4. Generates the Layer Sensitivity plot (Figure 1 in your doc)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.linear_probe import LinearProbe


def parse_args():
    parser = argparse.ArgumentParser(description="Train linear probes on collected latents")
    
    parser.add_argument(
        "--latents_dir",
        type=str,
        required=True,
        help="Directory containing collected latents (from collect_latents.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/probes",
        help="Output directory for trained probes"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Test split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Logistic regression regularization (C parameter)"
    )
    parser.add_argument(
        "--use_empirical",
        action="store_true",
        help="Use empirical FID/SSR measurements if available (requires quality_measurements.json)"
    )
    
    return parser.parse_args()


def load_latents(latents_dir: Path) -> dict:
    """Load all timestep latents from directory."""
    latents_dir = Path(latents_dir) / "latents"
    
    if not latents_dir.exists():
        # Try without /latents subdirectory
        latents_dir = Path(latents_dir).parent
    
    data = {}
    
    # Find all timestep files
    files = sorted(latents_dir.glob("timestep_*.npz"))
    
    if not files:
        raise FileNotFoundError(f"No timestep files found in {latents_dir}")
    
    print(f"Found {len(files)} timestep files", flush=True)
    
    for f in files:
        t = int(f.stem.split("_")[1])
        loaded = np.load(f)
        data[t] = {
            'X': loaded['X'],
            'y': loaded['y'],
        }
        
    return data


def train_probe(X: np.ndarray, y: np.ndarray, test_split: float, seed: int, C: float):
    """Train a single linear probe and return metrics."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    
    # Train logistic regression
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=seed,
        solver='lbfgs',
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_prob_test)
    
    return {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'auc': auc,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test,
        'y_prob': y_prob_test,
    }


def compute_sensitivity_scores(
    results: dict, 
    num_steps: int,
    latents_dir: Optional[Path] = None,
    use_empirical: bool = False,
) -> dict:
    """
    Compute Layer Sensitivity Scores.
    
    S_ℓ = Acc_ℓ × Qual_ℓ × Eff_ℓ
    
    If use_empirical=True and latents_dir is provided, attempts to measure:
    - Quality: FID between steered and unsteered images (1 - FID_norm)
    - Effectiveness: SSR improvement from steering experiments
    
    Otherwise, uses improved heuristics based on probe accuracy patterns.
    """
    sensitivity = {}
    
    # Try to load empirical measurements if available
    empirical_quality = {}
    empirical_effectiveness = {}
    
    if use_empirical and latents_dir is not None:
        try:
            empirical_quality, empirical_effectiveness = _load_empirical_measurements(
                latents_dir, num_steps
            )
            print(f"Loaded empirical measurements for {len(empirical_quality)} timesteps")
        except Exception as e:
            print(f"Warning: Could not load empirical measurements: {e}")
            print("Falling back to improved heuristics")
            use_empirical = False
    
    for t, r in results.items():
        acc = r['test_acc']
        
        # Quality preservation
        if use_empirical and t in empirical_quality:
            # Use measured FID (1 - FID_norm)
            qual = empirical_quality[t]
        else:
            # Improved heuristic: quality depends on probe accuracy
            # Higher accuracy = better concept separation = easier to steer without damage
            # Early timesteps (high t) have more noise, easier to modify
            # But we weight by accuracy: high accuracy timesteps are better for steering
            base_qual = 0.6 + 0.3 * (t / num_steps)  # Base: higher at early timesteps
            acc_weight = 0.1 * acc  # Bonus for high accuracy
            qual = min(1.0, base_qual + acc_weight)
        
        # Steering effectiveness
        if use_empirical and t in empirical_effectiveness:
            # Use measured SSR improvement
            eff = empirical_effectiveness[t]
        else:
            # Improved heuristic: effectiveness peaks where accuracy is high
            # Middle timesteps typically have best accuracy, so effectiveness peaks there
            mid = num_steps / 2
            sigma = num_steps / 3  # Slightly wider than before
            
            # Base bell curve
            base_eff = np.exp(-((t - mid) ** 2) / (2 * sigma ** 2))
            
            # Weight by accuracy: high accuracy timesteps are more effective
            acc_weight = 0.3 * acc
            eff = min(1.0, base_eff * 0.7 + acc_weight)
        
        # Combined score
        score = acc * qual * eff
        
        sensitivity[t] = {
            'accuracy': acc,
            'quality': qual,
            'effectiveness': eff,
            'score': score,
            'measured': use_empirical and (t in empirical_quality or t in empirical_effectiveness),
        }
    
    return sensitivity


def _load_empirical_measurements(latents_dir: Path, num_steps: int) -> tuple:
    """
    Load empirical quality and effectiveness measurements if available.
    
    Looks for:
    - quality_measurements.json: {timestep: 1 - FID_norm}
    - effectiveness_measurements.json: {timestep: SSR_improvement}
    
    Returns:
        (quality_dict, effectiveness_dict)
    """
    quality = {}
    effectiveness = {}
    
    quality_file = latents_dir / "quality_measurements.json"
    effectiveness_file = latents_dir / "effectiveness_measurements.json"
    
    if quality_file.exists():
        with open(quality_file, 'r') as f:
            quality_data = json.load(f)
            quality = {int(k): float(v) for k, v in quality_data.items()}
    
    if effectiveness_file.exists():
        with open(effectiveness_file, 'r') as f:
            effectiveness_data = json.load(f)
            effectiveness = {int(k): float(v) for k, v in effectiveness_data.items()}
    
    return quality, effectiveness


def find_optimal_window(sensitivity: dict, top_k: int = 5) -> tuple:
    """Find optimal intervention window from sensitivity scores."""
    # Sort by score
    sorted_t = sorted(sensitivity.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Get top-k timesteps
    top_timesteps = [t for t, _ in sorted_t[:top_k]]
    
    # Find contiguous window
    start = min(top_timesteps)
    end = max(top_timesteps)
    
    return start, end, top_timesteps


def plot_results(results: dict, sensitivity: dict, output_path: Path):
    """Generate visualization plots."""
    timesteps = sorted(results.keys())
    
    # Extract metrics
    train_accs = [results[t]['train_acc'] for t in timesteps]
    test_accs = [results[t]['test_acc'] for t in timesteps]
    aucs = [results[t]['auc'] for t in timesteps]
    
    sens_scores = [sensitivity[t]['score'] for t in timesteps]
    effectiveness = [sensitivity[t]['effectiveness'] for t in timesteps]
    quality = [sensitivity[t]['quality'] for t in timesteps]
    
    # Normalize timesteps to [0, 1] where 0=noise, 1=image
    max_t = max(timesteps)
    norm_t = [t / max_t for t in timesteps]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Probe Accuracy by Timestep
    ax1 = axes[0, 0]
    ax1.plot(norm_t, train_accs, 'b--', label='Train Accuracy', linewidth=2, alpha=0.7)
    ax1.plot(norm_t, test_accs, 'b-', label='Test Accuracy', linewidth=2)
    ax1.plot(norm_t, aucs, 'g-', label='AUC', linewidth=2)
    ax1.set_xlabel('Normalized Timestep (0=noise, 1=image)')
    ax1.set_ylabel('Score')
    ax1.set_title('Linear Probe Performance by Timestep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.0)
    
    # Plot 2: Layer Sensitivity Analysis (Figure 1 from your doc)
    ax2 = axes[0, 1]
    ax2.plot(norm_t, test_accs, 'b-', label='Probe Accuracy', linewidth=2)
    ax2.plot(norm_t, effectiveness, 'g--', label='Steering Effectiveness', linewidth=2)
    ax2.plot(norm_t, sens_scores, 'r-', label='Combined Score $S_\\ell$', linewidth=3)
    
    # Mark optimal point
    optimal_t = max(sensitivity.items(), key=lambda x: x[1]['score'])[0]
    optimal_score = sensitivity[optimal_t]['score']
    ax2.axvline(x=optimal_t/max_t, color='orange', linestyle=':', linewidth=2, 
                label=f'Optimal (t={optimal_t})')
    ax2.scatter([optimal_t/max_t], [optimal_score], color='orange', s=100, zorder=5)
    
    ax2.set_xlabel('Normalized Timestep')
    ax2.set_ylabel('Score')
    ax2.set_title('Layer Sensitivity Analysis')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Accuracy Heatmap Style
    ax3 = axes[1, 0]
    colors = plt.cm.RdYlGn(test_accs)
    bars = ax3.bar(timesteps, test_accs, color=colors)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Probe Accuracy per Timestep')
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax3.axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label='Target (85%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find optimal window
    start, end, top_t = find_optimal_window(sensitivity)
    
    summary_text = f"""
    ═══════════════════════════════════════════
    LINEAR PROBE ANALYSIS SUMMARY
    ═══════════════════════════════════════════
    
    Overall Metrics:
    ─────────────────────────────────────────
    • Mean Test Accuracy:  {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}
    • Mean AUC:            {np.mean(aucs):.3f} ± {np.std(aucs):.3f}
    • Best Accuracy:       {max(test_accs):.3f} (at t={timesteps[np.argmax(test_accs)]})
    • Best AUC:            {max(aucs):.3f} (at t={timesteps[np.argmax(aucs)]})
    
    Layer Sensitivity Analysis:
    ─────────────────────────────────────────
    • Optimal Timestep:    t={optimal_t}
    • Sensitivity Score:   {optimal_score:.3f}
    
    Recommended Intervention Window:
    ─────────────────────────────────────────
    • Start Step:          {start}
    • End Step:            {end}
    • Top-5 Timesteps:     {top_t}
    
    Separability Assessment:
    ─────────────────────────────────────────
    """
    
    # Add separability assessment
    if max(test_accs) > 0.85:
        summary_text += "    ✓ GOOD: Concepts are linearly separable (>85%)\n"
        summary_text += "    → Proceed to Phase 2 (PPO Training)"
    elif max(test_accs) > 0.70:
        summary_text += "    ⚠ MODERATE: Partial separability (70-85%)\n"
        summary_text += "    → Consider focusing on clearer concepts\n"
        summary_text += "    → Or try MLP probe instead of linear"
    else:
        summary_text += "    ✗ LOW: Poor separability (<70%)\n"
        summary_text += "    → Concepts may not be linearly separable\n"
        summary_text += "    → Try: MLP probe, different layers, coarser categories"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path / "probe_analysis.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "probe_analysis.pdf", bbox_inches='tight')
    print(f"Saved plots to {output_path}", flush=True)
    
    # Don't call plt.show() - just save the figures
    plt.close(fig)
    
    return fig


def save_pytorch_probes(results: dict, output_dir: Path, latent_dim: int):
    """Convert sklearn probes to PyTorch and save."""
    probes_dir = output_dir / "pytorch"
    probes_dir.mkdir(exist_ok=True)
    
    for t, r in results.items():
        sklearn_model = r['model']
        probe = LinearProbe.from_sklearn(sklearn_model, latent_dim)
        torch.save(probe.state_dict(), probes_dir / f"probe_t{t:02d}.pt")
    
    print(f"Saved PyTorch probes to {probes_dir}")


def main():
    import sys
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60, flush=True)
    print("LINEAR PROBE TRAINING", flush=True)
    print("="*60, flush=True)
    print(f"Latents: {args.latents_dir}", flush=True)
    print(f"Output: {run_dir}", flush=True)
    
    # Load latents
    print("\n--- Loading Latents ---", flush=True)
    data = load_latents(args.latents_dir)
    
    num_steps = max(data.keys())
    sample_X = data[0]['X']
    latent_dim = sample_X.shape[1]
    num_samples = sample_X.shape[0]
    
    print(f"Timesteps: 0 to {num_steps}", flush=True)
    print(f"Latent dimension: {latent_dim}", flush=True)
    print(f"Total samples: {num_samples}", flush=True)
    print(f"Label distribution: {np.bincount(data[0]['y'])}", flush=True)
    
    # Train probes at each timestep
    print("\n--- Training Probes ---", flush=True)
    results = {}
    
    for t in tqdm(sorted(data.keys()), desc="Training probes"):
        X, y = data[t]['X'], data[t]['y']
        results[t] = train_probe(
            X, y, 
            test_split=args.test_split,
            seed=args.seed,
            C=args.regularization,
        )
    
    # Print accuracy summary
    print("\n--- Probe Accuracies ---", flush=True)
    for t in sorted(results.keys()):
        r = results[t]
        print(f"  t={t:2d}: Train={r['train_acc']:.3f}, Test={r['test_acc']:.3f}, AUC={r['auc']:.3f}", flush=True)
    
    # Compute sensitivity scores
    print("\n--- Computing Sensitivity Scores ---")
    latents_dir = Path(args.latents_dir)
    sensitivity = compute_sensitivity_scores(
        results, 
        num_steps,
        latents_dir=latents_dir,
        use_empirical=args.use_empirical,
    )
    
    # Report measurement status
    measured_count = sum(1 for s in sensitivity.values() if s.get('measured', False))
    if measured_count > 0:
        print(f"Using empirical measurements for {measured_count}/{len(sensitivity)} timesteps")
    else:
        print("Using improved heuristics (run with --use_empirical to enable empirical measurements)")
    
    start, end, top_t = find_optimal_window(sensitivity)
    print(f"Optimal intervention window: [{start}, {end}]")
    print(f"Top timesteps: {top_t}")
    
    # Save results
    print("\n--- Saving Results ---")
    
    # Save metrics
    metrics = {
        t: {
            'train_acc': r['train_acc'],
            'test_acc': r['test_acc'],
            'auc': r['auc'],
        }
        for t, r in results.items()
    }
    
    with open(run_dir / "probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save sensitivity scores
    sens_data = {
        str(t): {
            'accuracy': s['accuracy'],
            'quality': s['quality'],
            'effectiveness': s['effectiveness'],
            'score': s['score'],
        }
        for t, s in sensitivity.items()
    }
    sens_data['optimal_window'] = {'start': start, 'end': end, 'top_timesteps': top_t}
    
    with open(run_dir / "sensitivity_scores.json", "w") as f:
        json.dump(sens_data, f, indent=2)
    
    # Save PyTorch probes
    save_pytorch_probes(results, run_dir, latent_dim)
    
    # Generate plots
    print("\n--- Generating Plots ---")
    plot_results(results, sensitivity, run_dir)
    
    print("\n" + "="*60)
    print("✓ PROBE TRAINING COMPLETE")
    print("="*60)
    print(f"Output: {run_dir}")
    print(f"Best accuracy: {max(r['test_acc'] for r in results.values()):.3f}")
    print(f"Recommended intervention window: steps {start} to {end}")
    
    # Final recommendation
    best_acc = max(r['test_acc'] for r in results.values())
    if best_acc > 0.85:
        print("\n✓ Linear separability confirmed! Proceed to Phase 2.")
    elif best_acc > 0.70:
        print("\n⚠ Moderate separability. Consider adjusting categories or using MLP.")
    else:
        print("\n✗ Poor separability. Review data or try alternative approaches.")
    
    return run_dir


if __name__ == "__main__":
    main()

