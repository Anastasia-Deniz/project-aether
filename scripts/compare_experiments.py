"""
Compare and visualize results from multiple experiments.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --experiments exp_lambda_0.5 exp_lambda_0.8
    python scripts/compare_experiments.py --plot  # Generate plots
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_experiment_history(experiment_name: str) -> Optional[Dict]:
    """Load training history for an experiment."""
    output_dir = Path("outputs/ppo")
    if not output_dir.exists():
        return None
    
    exp_dirs = list(output_dir.glob(f"{experiment_name}_*"))
    if not exp_dirs:
        return None
    
    # Get latest run
    latest_dir = max(exp_dirs, key=lambda p: p.stat().st_mtime)
    history_file = latest_dir / "training_history.json"
    
    if not history_file.exists():
        return None
    
    with open(history_file) as f:
        history = json.load(f)
    
    return {
        "experiment_name": experiment_name,
        "run_dir": str(latest_dir),
        "history": history,
    }


def compare_experiments(experiment_names: Optional[List[str]] = None):
    """Compare multiple experiments."""
    if experiment_names is None:
        # Find all experiments
        output_dir = Path("outputs/ppo")
        if not output_dir.exists():
            print("❌ No experiments found in outputs/ppo/")
            return
        
        exp_dirs = list(output_dir.glob("exp_*"))
        experiment_names = list(set(d.name.split("_")[0] + "_" + "_".join(d.name.split("_")[1:-1]) 
                                   for d in exp_dirs if d.is_dir()))
        experiment_names = sorted(experiment_names)
    
    print(f"\n📊 Comparing {len(experiment_names)} experiments\n")
    
    results = []
    for exp_name in experiment_names:
        data = load_experiment_history(exp_name)
        if data:
            history = data["history"]
            
            result = {
                "experiment_name": exp_name,
                "run_dir": data["run_dir"],
            }
            
            # Extract metrics
            if "rewards" in history and history["rewards"]:
                rewards = history["rewards"]
                result["final_reward"] = rewards[-1]
                result["mean_reward"] = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                result["max_reward"] = np.max(rewards)
                result["min_reward"] = np.min(rewards)
                result["num_updates"] = len(rewards)
            
            if "policy_loss" in history and history["policy_loss"]:
                policy_losses = history["policy_loss"]
                result["final_policy_loss"] = policy_losses[-1]
                result["mean_policy_loss"] = np.mean(policy_losses[-10:]) if len(policy_losses) >= 10 else np.mean(policy_losses)
            
            if "value_loss" in history and history["value_loss"]:
                value_losses = history["value_loss"]
                result["final_value_loss"] = value_losses[-1]
            
            results.append(result)
    
    if not results:
        print("❌ No completed experiments found")
        return
    
    # Sort by final reward
    results.sort(key=lambda x: x.get("final_reward", -float("inf")), reverse=True)
    
    # Print comparison table
    print(f"{'Experiment':<35} {'Final Reward':<15} {'Mean Reward':<15} {'Policy Loss':<15} {'Updates':<10}")
    print("-" * 100)
    
    for r in results:
        exp_name = r["experiment_name"]
        final_reward = r.get("final_reward", "N/A")
        mean_reward = r.get("mean_reward", "N/A")
        policy_loss = r.get("final_policy_loss", "N/A")
        num_updates = r.get("num_updates", "N/A")
        
        if isinstance(final_reward, (int, float)):
            final_reward = f"{final_reward:.4f}"
        if isinstance(mean_reward, (int, float)):
            mean_reward = f"{mean_reward:.4f}"
        if isinstance(policy_loss, (int, float)):
            policy_loss = f"{policy_loss:.4f}"
        
        print(f"{exp_name:<35} {str(final_reward):<15} {str(mean_reward):<15} {str(policy_loss):<15} {str(num_updates):<10}")
    
    # Print best experiment
    if results and results[0].get("final_reward") is not None:
        best = results[0]
        print(f"\n🏆 Best Experiment: {best['experiment_name']}")
        print(f"   Final Reward: {best.get('final_reward', 'N/A'):.4f}")
        print(f"   Mean Reward: {best.get('mean_reward', 'N/A'):.4f}")
        print(f"   Run Directory: {best.get('run_dir', 'N/A')}")
    
    return results


def plot_experiments(experiment_names: List[str], output_dir: Path = Path("outputs/visualizations")):
    """Plot training curves for multiple experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for exp_name in experiment_names:
        data = load_experiment_history(exp_name)
        if not data or "history" not in data:
            continue
        
        history = data["history"]
        
        # Plot rewards
        if "rewards" in history and history["rewards"]:
            axes[0, 0].plot(history["rewards"], label=exp_name, alpha=0.7)
        
        # Plot policy loss
        if "policy_loss" in history and history["policy_loss"]:
            axes[0, 1].plot(history["policy_loss"], label=exp_name, alpha=0.7)
        
        # Plot value loss
        if "value_loss" in history and history["value_loss"]:
            axes[1, 0].plot(history["value_loss"], label=exp_name, alpha=0.7)
        
        # Plot entropy
        if "entropy" in history and history["entropy"]:
            axes[1, 1].plot(history["entropy"], label=exp_name, alpha=0.7)
    
    axes[0, 0].set_title("Rewards")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title("Entropy")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / "experiments_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot saved to: {plot_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiments to compare (default: all)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    
    args = parser.parse_args()
    
    # Compare experiments
    results = compare_experiments(args.experiments)
    
    # Generate plots if requested
    if args.plot and results:
        experiment_names = [r["experiment_name"] for r in results]
        plot_experiments(experiment_names)
        print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
