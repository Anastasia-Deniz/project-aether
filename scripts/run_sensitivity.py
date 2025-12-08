"""
Project Aether - Layer Sensitivity Analysis Script
Phase 1: Identify optimal intervention timesteps for steering.

This script:
1. Loads trained probe results from train_probes.py
2. Estimates steering effectiveness at each timestep
3. Computes layer sensitivity scores (Equation 2 from the paper)
4. Generates Figure 1: Layer Sensitivity Analysis plot
5. Outputs the recommended intervention window

Usage:
    python scripts/run_sensitivity.py --probes_dir ./checkpoints/probes/run_XXXXX
    
    # With custom effectiveness estimation
    python scripts/run_sensitivity.py --probes_dir ./checkpoints/probes/run_XXXXX --estimate_effectiveness
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class LayerSensitivityResult:
    """Result of layer sensitivity analysis for one timestep."""
    timestep: int
    normalized_t: float  # 0 = noise, 1 = image
    probe_accuracy: float
    probe_auc: float
    quality_preservation: float  # Estimated
    steering_effectiveness: float  # Estimated
    sensitivity_score: float  # Combined score S_ℓ


def parse_args():
    parser = argparse.ArgumentParser(description="Run layer sensitivity analysis")
    
    parser.add_argument(
        "--probes_dir",
        type=str,
        required=True,
        help="Directory containing trained probes (from train_probes.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/sensitivity",
        help="Output directory for sensitivity analysis"
    )
    parser.add_argument(
        "--estimate_effectiveness",
        action="store_true",
        help="Use heuristic to estimate steering effectiveness"
    )
    parser.add_argument(
        "--effectiveness_peak",
        type=float,
        default=0.5,
        help="Normalized timestep where effectiveness peaks (0-1)"
    )
    parser.add_argument(
        "--effectiveness_width",
        type=float,
        default=0.25,
        help="Width of effectiveness bell curve"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top timesteps to include in intervention window"
    )
    
    return parser.parse_args()


def load_probe_metrics(probes_dir: Path) -> Dict:
    """Load probe metrics from JSON file."""
    metrics_file = probes_dir / "probe_metrics.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Probe metrics not found: {metrics_file}")
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    # Convert string keys to int
    return {int(k): v for k, v in metrics.items()}


def estimate_steering_effectiveness(
    num_timesteps: int,
    peak: float = 0.5,
    width: float = 0.25,
) -> Dict[int, float]:
    """
    Estimate steering effectiveness using a bell curve heuristic.
    
    Based on MACE's Semantic Single Boundary (SSB) concept:
    - Early timesteps: global structure being formed, steering affects everything
    - Middle timesteps: optimal for concept-specific steering
    - Late timesteps: details crystallized, steering has limited effect
    
    Args:
        num_timesteps: Total number of diffusion steps
        peak: Normalized timestep where effectiveness peaks (0=noise, 1=image)
        width: Width of the bell curve
        
    Returns:
        Dict mapping timestep -> effectiveness score (0-1)
    """
    effectiveness = {}
    
    for t in range(num_timesteps + 1):
        # Convert to normalized time (0 = noise, 1 = image)
        norm_t = t / num_timesteps
        
        # Bell curve centered at peak
        eff = np.exp(-((norm_t - peak) ** 2) / (2 * width ** 2))
        effectiveness[t] = float(eff)
    
    return effectiveness


def estimate_quality_preservation(num_timesteps: int) -> Dict[int, float]:
    """
    Estimate quality preservation at each timestep.
    
    Heuristic based on:
    - Early timesteps (more noise): easier to steer without quality loss
    - Late timesteps (more structure): steering may damage quality
    
    Args:
        num_timesteps: Total number of diffusion steps
        
    Returns:
        Dict mapping timestep -> quality preservation score (0-1)
    """
    quality = {}
    
    for t in range(num_timesteps + 1):
        # Normalized time (0 = noise, 1 = image)
        norm_t = t / num_timesteps
        
        # Quality preservation is higher at early timesteps
        # Sigmoid-like decay toward late timesteps
        qual = 0.7 + 0.3 * (1 - norm_t)
        quality[t] = float(qual)
    
    return quality


def compute_sensitivity_scores(
    probe_metrics: Dict,
    effectiveness: Dict[int, float],
    quality: Dict[int, float],
) -> List[LayerSensitivityResult]:
    """
    Compute Layer Sensitivity Score from Equation 2:
    
    S_ℓ = Acc_ℓ × (1 - FID_norm_ℓ) × ΔSSR_ℓ
    
    We approximate:
    - Acc_ℓ: probe test accuracy
    - (1 - FID_norm_ℓ): quality preservation estimate
    - ΔSSR_ℓ: steering effectiveness estimate
    """
    results = []
    num_timesteps = max(probe_metrics.keys())
    
    for t, metrics in probe_metrics.items():
        acc = metrics['test_acc']
        qual = quality.get(t, 0.5)
        eff = effectiveness.get(t, 0.5)
        
        # Combined sensitivity score
        score = acc * qual * eff
        
        results.append(LayerSensitivityResult(
            timestep=t,
            normalized_t=t / num_timesteps,
            probe_accuracy=acc,
            probe_auc=metrics.get('auc', acc),
            quality_preservation=qual,
            steering_effectiveness=eff,
            sensitivity_score=score,
        ))
    
    # Sort by sensitivity score (descending)
    results.sort(key=lambda x: x.sensitivity_score, reverse=True)
    
    return results


def find_optimal_window(
    results: List[LayerSensitivityResult],
    top_k: int = 5,
) -> Tuple[int, int, List[int]]:
    """
    Find optimal intervention window from sensitivity results.
    
    Args:
        results: Sorted sensitivity results (best first)
        top_k: Number of top timesteps to consider
        
    Returns:
        (start_step, end_step, top_timesteps)
    """
    # Get top-k timesteps by sensitivity
    top_timesteps = [r.timestep for r in results[:top_k]]
    
    # Find contiguous window
    start = min(top_timesteps)
    end = max(top_timesteps)
    
    return start, end, sorted(top_timesteps)


def plot_sensitivity_analysis(
    results: List[LayerSensitivityResult],
    optimal_start: int,
    optimal_end: int,
    output_path: Path,
) -> plt.Figure:
    """Generate the Layer Sensitivity Analysis figure (Figure 1 from paper)."""
    
    # Sort by timestep for plotting
    sorted_results = sorted(results, key=lambda x: x.timestep)
    
    timesteps = [r.timestep for r in sorted_results]
    norm_t = [r.normalized_t for r in sorted_results]
    accuracies = [r.probe_accuracy for r in sorted_results]
    effectiveness = [r.steering_effectiveness for r in sorted_results]
    quality = [r.quality_preservation for r in sorted_results]
    combined = [r.sensitivity_score for r in sorted_results]
    
    # Find optimal point
    optimal_idx = np.argmax(combined)
    optimal_score = combined[optimal_idx]
    optimal_t = norm_t[optimal_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Probe Accuracy
    ax1 = axes[0, 0]
    ax1.plot(norm_t, accuracies, 'b-', linewidth=2, label='Test Accuracy')
    ax1.fill_between(norm_t, 0.5, accuracies, alpha=0.2, color='blue')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target (85%)')
    ax1.set_xlabel('Normalized Timestep (0=noise, 1=image)')
    ax1.set_ylabel('Probe Accuracy')
    ax1.set_title('Linear Probe Accuracy vs Timestep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.4, 1.0)
    
    # Plot 2: Layer Sensitivity Analysis (main figure)
    ax2 = axes[0, 1]
    ax2.plot(norm_t, accuracies, 'b-', linewidth=2, label='Probe Accuracy')
    ax2.plot(norm_t, effectiveness, 'g--', linewidth=2, label='Steering Effectiveness')
    ax2.plot(norm_t, quality, 'c:', linewidth=2, label='Quality Preservation')
    ax2.plot(norm_t, combined, 'r-', linewidth=3, label='Combined Score $S_\\ell$')
    
    # Mark optimal point
    ax2.axvline(x=optimal_t, color='orange', linestyle=':', linewidth=2)
    ax2.scatter([optimal_t], [optimal_score], color='orange', s=150, zorder=5, 
                label=f'Optimal (t={optimal_t:.2f})')
    
    # Shade intervention window
    num_steps = max(timesteps)
    window_start = optimal_start / num_steps
    window_end = optimal_end / num_steps
    ax2.axvspan(window_start, window_end, alpha=0.1, color='green', 
                label=f'Intervention Window')
    
    ax2.set_xlabel('Normalized Timestep (0=noise, 1=image)')
    ax2.set_ylabel('Score')
    ax2.set_title('Layer Sensitivity Analysis (Equation 2)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Component Contributions
    ax3 = axes[1, 0]
    width = 0.8 / 3
    x = np.arange(len(timesteps))
    
    ax3.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
    ax3.bar(x, effectiveness, width, label='Effectiveness', alpha=0.8)
    ax3.bar(x + width, quality, width, label='Quality', alpha=0.8)
    
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Score')
    ax3.set_title('Component Scores per Timestep')
    ax3.set_xticks(x)
    ax3.set_xticklabels(timesteps)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Best results
    best_results = sorted(results, key=lambda x: x.sensitivity_score, reverse=True)[:5]
    
    summary_text = f"""
    ═══════════════════════════════════════════════════════════
    LAYER SENSITIVITY ANALYSIS RESULTS
    ═══════════════════════════════════════════════════════════
    
    Recommended Intervention Window:
    ─────────────────────────────────────────────────────────
    • Start Step:        {optimal_start}
    • End Step:          {optimal_end}
    • Window Size:       {optimal_end - optimal_start + 1} steps
    
    Optimal Timestep:
    ─────────────────────────────────────────────────────────
    • Timestep:          {best_results[0].timestep}
    • Normalized:        {best_results[0].normalized_t:.3f}
    • Sensitivity:       {best_results[0].sensitivity_score:.4f}
    
    Top 5 Timesteps by Sensitivity Score:
    ─────────────────────────────────────────────────────────
"""
    
    for i, r in enumerate(best_results):
        summary_text += f"    {i+1}. t={r.timestep:2d} (S={r.sensitivity_score:.3f}, "
        summary_text += f"Acc={r.probe_accuracy:.3f}, Eff={r.steering_effectiveness:.3f})\n"
    
    summary_text += f"""
    Key Insights:
    ─────────────────────────────────────────────────────────
    • Max Probe Accuracy:    {max(accuracies):.3f} (at t={timesteps[np.argmax(accuracies)]})
    • Mean Accuracy:         {np.mean(accuracies):.3f}
    • Optimal is at:         {optimal_t:.1%} of generation process
    
    Config for train_ppo.yaml:
    ─────────────────────────────────────────────────────────
    intervention:
      start_timestep: {optimal_start}
      end_timestep: {optimal_end}
    """
    
    ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(output_path / "sensitivity_analysis.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "sensitivity_analysis.pdf", bbox_inches='tight')
    
    return fig


def main():
    args = parse_args()
    
    # Setup paths
    probes_dir = Path(args.probes_dir)
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LAYER SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Probes directory: {probes_dir}")
    print(f"Output directory: {run_dir}")
    
    # Load probe metrics
    print("\n--- Loading Probe Metrics ---")
    try:
        probe_metrics = load_probe_metrics(probes_dir)
        num_timesteps = max(probe_metrics.keys())
        print(f"Loaded metrics for {len(probe_metrics)} timesteps (0 to {num_timesteps})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_probes.py first to generate probe metrics.")
        return None
    
    # Check if sensitivity scores already exist
    existing_sensitivity = probes_dir / "sensitivity_scores.json"
    if existing_sensitivity.exists():
        print(f"\nFound existing sensitivity scores: {existing_sensitivity}")
        with open(existing_sensitivity, 'r') as f:
            existing = json.load(f)
        if 'optimal_window' in existing:
            print(f"  Existing window: [{existing['optimal_window']['start']}, {existing['optimal_window']['end']}]")
    
    # Estimate effectiveness and quality
    print("\n--- Estimating Component Scores ---")
    effectiveness = estimate_steering_effectiveness(
        num_timesteps,
        peak=args.effectiveness_peak,
        width=args.effectiveness_width,
    )
    quality = estimate_quality_preservation(num_timesteps)
    
    print(f"Effectiveness peak at normalized t={args.effectiveness_peak:.2f}")
    print(f"Effectiveness width: {args.effectiveness_width:.2f}")
    
    # Compute sensitivity scores
    print("\n--- Computing Sensitivity Scores ---")
    results = compute_sensitivity_scores(probe_metrics, effectiveness, quality)
    
    # Find optimal window
    start, end, top_t = find_optimal_window(results, top_k=args.top_k)
    
    print(f"\nOptimal intervention window: steps {start} to {end}")
    print(f"Top {args.top_k} timesteps: {top_t}")
    
    # Print top results
    print("\n--- Top Sensitivity Scores ---")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. t={r.timestep:2d}: S={r.sensitivity_score:.4f} "
              f"(Acc={r.probe_accuracy:.3f}, Eff={r.steering_effectiveness:.3f}, "
              f"Qual={r.quality_preservation:.3f})")
    
    # Save results
    print("\n--- Saving Results ---")
    
    # Save detailed results
    results_data = {
        str(r.timestep): {
            'normalized_t': r.normalized_t,
            'probe_accuracy': r.probe_accuracy,
            'probe_auc': r.probe_auc,
            'quality_preservation': r.quality_preservation,
            'steering_effectiveness': r.steering_effectiveness,
            'sensitivity_score': r.sensitivity_score,
        }
        for r in results
    }
    results_data['optimal_window'] = {
        'start': start,
        'end': end,
        'top_timesteps': top_t,
    }
    results_data['config'] = {
        'effectiveness_peak': args.effectiveness_peak,
        'effectiveness_width': args.effectiveness_width,
        'top_k': args.top_k,
    }
    
    with open(run_dir / "sensitivity_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Generate plots
    print("\n--- Generating Plots ---")
    fig = plot_sensitivity_analysis(results, start, end, run_dir)
    plt.show()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("✓ SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output: {run_dir}")
    print(f"\nRecommended PPO config updates:")
    print(f"  intervention_start: {start}")
    print(f"  intervention_end: {end}")
    
    return run_dir


if __name__ == "__main__":
    main()

