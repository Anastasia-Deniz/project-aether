"""
Compare results from multiple PPO training experiments.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --experiments exp_lambda_0.3 exp_lambda_0.5
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def load_training_history(exp_dir: Path) -> Optional[Dict]:
    """Load training history from experiment directory."""
    history_file = exp_dir / "training_history.json"
    if not history_file.exists():
        return None
    
    with open(history_file) as f:
        return json.load(f)

def get_experiment_dirs(experiment_names: Optional[List[str]] = None) -> Dict[str, Path]:
    """Get experiment directories."""
    output_dir = Path("outputs/ppo")
    if not output_dir.exists():
        return {}
    
    experiments = {}
    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Extract experiment name (format: exp_name_timestamp)
        parts = exp_dir.name.split("_")
        if len(parts) < 2:
            continue
        
        # Try to find experiment name (everything before last timestamp-like part)
        exp_name = "_".join(parts[:-1]) if len(parts) > 2 else parts[0]
        
        # Filter by requested experiments
        if experiment_names and exp_name not in experiment_names:
            continue
        
        # Only include if has training history
        if (exp_dir / "training_history.json").exists():
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(exp_dir)
    
    # For each experiment, get the most recent run
    result = {}
    for exp_name, dirs in experiments.items():
        # Sort by modification time, get most recent
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)
        result[exp_name] = most_recent
    
    return result

def compute_metrics(history: Dict) -> Dict:
    """Compute summary metrics from training history."""
    metrics = {}
    
    # Final values
    if history.get("rewards"):
        rewards = history["rewards"]
        metrics["final_reward"] = rewards[-1]
        metrics["mean_reward"] = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        metrics["max_reward"] = np.max(rewards)
        metrics["reward_std"] = np.std(rewards[-10:]) if len(rewards) >= 10 else np.std(rewards)
    
    if history.get("policy_losses"):
        policy_losses = history["policy_losses"]
        metrics["final_policy_loss"] = policy_losses[-1]
        metrics["mean_policy_loss"] = np.mean(policy_losses[-10:]) if len(policy_losses) >= 10 else np.mean(policy_losses)
    
    if history.get("value_losses"):
        value_losses = history["value_losses"]
        metrics["final_value_loss"] = value_losses[-1]
        metrics["mean_value_loss"] = np.mean(value_losses[-10:]) if len(value_losses) >= 10 else np.mean(value_losses)
    
    if history.get("entropies"):
        entropies = history["entropies"]
        metrics["final_entropy"] = entropies[-1]
        metrics["mean_entropy"] = np.mean(entropies[-10:]) if len(entropies) >= 10 else np.mean(entropies)
    
    # Training progress
    metrics["total_updates"] = len(history.get("rewards", []))
    
    return metrics

def print_comparison(experiments: Dict[str, Path]):
    """Print comparison table of experiments."""
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON")
    print("="*100)
    
    all_metrics = {}
    
    for exp_name, exp_dir in sorted(experiments.items()):
        history = load_training_history(exp_dir)
        if not history:
            continue
        
        metrics = compute_metrics(history)
        all_metrics[exp_name] = metrics
    
    if not all_metrics:
        print("No training histories found!")
        return
    
    # Print table
    print(f"\n{'Experiment':<25} {'Final Reward':<15} {'Mean Reward':<15} {'Policy Loss':<15} {'Value Loss':<15} {'Updates':<10}")
    print("-" * 100)
    
    for exp_name, metrics in sorted(all_metrics.items()):
        print(
            f"{exp_name:<25} "
            f"{metrics.get('final_reward', 0):>14.4f} "
            f"{metrics.get('mean_reward', 0):>14.4f} "
            f"{metrics.get('final_policy_loss', 0):>14.4f} "
            f"{metrics.get('final_value_loss', 0):>14.4f} "
            f"{metrics.get('total_updates', 0):>9}"
        )
    
    # Find best experiments
    print("\n" + "="*100)
    print("BEST EXPERIMENTS")
    print("="*100)
    
    if all_metrics:
        # Best final reward
        best_reward = max(all_metrics.items(), key=lambda x: x[1].get('final_reward', -float('inf')))
        print(f"Best Final Reward: {best_reward[0]} ({best_reward[1].get('final_reward', 0):.4f})")
        
        # Best mean reward
        best_mean = max(all_metrics.items(), key=lambda x: x[1].get('mean_reward', -float('inf')))
        print(f"Best Mean Reward: {best_mean[0]} ({best_mean[1].get('mean_reward', 0):.4f})")
        
        # Lowest policy loss
        best_policy = min(all_metrics.items(), key=lambda x: x[1].get('final_policy_loss', float('inf')))
        print(f"Lowest Policy Loss: {best_policy[0]} ({best_policy[1].get('final_policy_loss', 0):.4f})")
        
        # Lowest value loss
        best_value = min(all_metrics.items(), key=lambda x: x[1].get('final_value_loss', float('inf')))
        print(f"Lowest Value Loss: {best_value[0]} ({best_value[1].get('final_value_loss', 0):.4f})")

def main():
    parser = argparse.ArgumentParser(description="Compare PPO training experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiments to compare (default: all)",
    )
    
    args = parser.parse_args()
    
    experiments = get_experiment_dirs(args.experiments)
    
    if not experiments:
        print("No experiments found!")
        print("Run experiments first with: python scripts/run_experiments.py")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp_name in sorted(experiments.keys()):
        print(f"  - {exp_name}")
    
    print_comparison(experiments)

if __name__ == "__main__":
    main()

