"""
Run multiple PPO training experiments with different hyperparameters.

This script runs all experiments sequentially and saves results for comparison.
After running, it can generate a comparison report.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --experiments exp_lambda_0.3 exp_lambda_0.5
    python scripts/run_experiments.py --skip-completed
    python scripts/run_experiments.py --compare  # Compare completed experiments
    python scripts/run_experiments.py --list  # List all experiments
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_experiment_configs():
    """Get all experiment config files."""
    exp_dir = Path("configs/experiments")
    if not exp_dir.exists():
        return []
    
    configs = sorted(exp_dir.glob("*.yaml"))
    return [c.stem for c in configs]


def load_experiment_config(config_name: str) -> dict:
    """Load experiment config and extract key parameters."""
    config_path = f"configs/experiments/{config_name}.yaml"
    if not Path(config_path).exists():
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def check_experiment_completed(experiment_name: str) -> bool:
    """Check if experiment has completed (has final_policy.pt)."""
    output_dir = Path("outputs/ppo")
    if not output_dir.exists():
        return False
    
    # Look for experiment directory
    exp_dirs = list(output_dir.glob(f"{experiment_name}_*"))
    if not exp_dirs:
        return False
    
    # Check if final policy exists
    for exp_dir in exp_dirs:
        final_policy = exp_dir / "final_policy.pt"
        if final_policy.exists():
            return True
    
    return False


def get_experiment_results(experiment_name: str) -> dict:
    """Extract results from completed experiment."""
    output_dir = Path("outputs/ppo")
    if not output_dir.exists():
        return {}
    
    exp_dirs = list(output_dir.glob(f"{experiment_name}_*"))
    if not exp_dirs:
        return {}
    
    # Get latest run
    latest_dir = max(exp_dirs, key=lambda p: p.stat().st_mtime)
    
    results = {
        "experiment_name": experiment_name,
        "run_dir": str(latest_dir),
        "final_policy_exists": (latest_dir / "final_policy.pt").exists(),
    }
    
    # Load training history if available
    history_file = latest_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
            if history.get("rewards"):
                results["final_reward"] = history["rewards"][-1] if history["rewards"] else None
                results["mean_reward"] = sum(history["rewards"][-10:]) / min(10, len(history["rewards"])) if history["rewards"] else None
            if history.get("policy_loss"):
                results["final_policy_loss"] = history["policy_loss"][-1] if history["policy_loss"] else None
    
    return results


def run_experiment(config_name: str, skip_completed: bool = False) -> bool:
    """Run a single experiment."""
    config_path = f"configs/experiments/{config_name}.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    # Check if already completed
    if skip_completed:
        config = load_experiment_config(config_name)
        exp_name = config.get("training", {}).get("experiment_name", config_name)
        if check_experiment_completed(exp_name):
            print(f"⏭️  Skipping {config_name} (already completed)")
            return True
    
    print("\n" + "="*80)
    print(f"🚀 Running Experiment: {config_name}")
    print("="*80)
    
    # Show experiment parameters
    config = load_experiment_config(config_name)
    if config:
        print("\nExperiment Parameters:")
        if "ppo" in config:
            ppo = config["ppo"]
            print(f"  Total timesteps: {ppo.get('total_timesteps', 'N/A'):,}")
            print(f"  Learning rate: {ppo.get('learning_rate', 'N/A')}")
            print(f"  Batch size: {ppo.get('batch_size', 'N/A')}")
            print(f"  Epochs: {ppo.get('n_epochs', 'N/A')}")
        if "reward" in config:
            print(f"  Lambda transport: {config['reward'].get('lambda_transport', 'N/A')}")
        if "env" in config:
            print(f"  Inference steps: {config['env'].get('num_inference_steps', 'N/A')}")
            print(f"  Intervention: [{config['env'].get('intervention_start', 'N/A')}, {config['env'].get('intervention_end', 'N/A')}]")
    
    print("="*80)
    
    # Run training
    cmd = [
        sys.executable,
        "scripts/train_ppo.py",
        "--config", config_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent.parent,
            capture_output=False,  # Show output in real-time
        )
        
        print(f"\n✅ Experiment {config_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment {config_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment {config_name} interrupted by user")
        return False


def compare_experiments():
    """Compare results from all completed experiments."""
    print("\n" + "="*80)
    print("📊 COMPARING EXPERIMENTS")
    print("="*80)
    
    experiments = get_experiment_configs()
    if not experiments:
        print("❌ No experiments found")
        return
    
    results = []
    for exp in experiments:
        config = load_experiment_config(exp)
        exp_name = config.get("training", {}).get("experiment_name", exp)
        
        if check_experiment_completed(exp_name):
            result = get_experiment_results(exp_name)
            result["config"] = config
            results.append(result)
    
    if not results:
        print("❌ No completed experiments found")
        return
    
    # Sort by final reward (if available)
    results.sort(key=lambda x: x.get("final_reward", -float("inf")), reverse=True)
    
    print(f"\nFound {len(results)} completed experiments:\n")
    
    # Print comparison table
    print(f"{'Experiment':<30} {'Final Reward':<15} {'Mean Reward':<15} {'Policy Loss':<15}")
    print("-" * 80)
    
    for r in results:
        exp_name = r["experiment_name"]
        final_reward = r.get("final_reward", "N/A")
        mean_reward = r.get("mean_reward", "N/A")
        policy_loss = r.get("final_policy_loss", "N/A")
        
        if isinstance(final_reward, (int, float)):
            final_reward = f"{final_reward:.4f}"
        if isinstance(mean_reward, (int, float)):
            mean_reward = f"{mean_reward:.4f}"
        if isinstance(policy_loss, (int, float)):
            policy_loss = f"{policy_loss:.4f}"
        
        print(f"{exp_name:<30} {str(final_reward):<15} {str(mean_reward):<15} {str(policy_loss):<15}")
    
    # Save comparison
    comparison_file = Path("outputs/experiments_comparison.json")
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "experiments": results,
    }
    
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Comparison saved to: {comparison_file}")
    
    # Print best experiment
    if results and results[0].get("final_reward") is not None:
        best = results[0]
        print(f"\n🏆 Best Experiment: {best['experiment_name']}")
        print(f"   Final Reward: {best.get('final_reward', 'N/A')}")
        print(f"   Run Directory: {best.get('run_dir', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run PPO training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python scripts/run_experiments.py --list
  
  # Run all experiments
  python scripts/run_experiments.py
  
  # Run specific experiments
  python scripts/run_experiments.py --experiments exp_lambda_0.5 exp_lambda_0.8
  
  # Skip already completed experiments
  python scripts/run_experiments.py --skip-completed
  
  # Compare completed experiments
  python scripts/run_experiments.py --compare
        """
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip experiments that have already completed",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results from completed experiments",
    )
    
    args = parser.parse_args()
    
    # Compare mode
    if args.compare:
        compare_experiments()
        return
    
    # List experiments
    if args.list:
        configs = get_experiment_configs()
        print("Available experiments:")
        print(f"{'Experiment':<40} {'Status':<15} {'Parameters'}")
        print("-" * 100)
        
        for config_name in configs:
            config = load_experiment_config(config_name)
            exp_name = config.get("training", {}).get("experiment_name", config_name)
            status = "✅ Completed" if check_experiment_completed(exp_name) else "⏳ Pending"
            
            # Extract key parameters
            params = []
            if "ppo" in config:
                ppo = config["ppo"]
                params.append(f"timesteps={ppo.get('total_timesteps', 'N/A')}")
                params.append(f"lr={ppo.get('learning_rate', 'N/A')}")
            if "reward" in config:
                params.append(f"λ={config['reward'].get('lambda_transport', 'N/A')}")
            
            params_str = ", ".join(params)
            print(f"{config_name:<40} {status:<15} {params_str}")
        return
    
    # Get experiments to run
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = get_experiment_configs()
    
    if not experiments:
        print("❌ No experiments found in configs/experiments/")
        print("   Create experiment configs first!")
        return
    
    print(f"\n📋 Running {len(experiments)} experiments")
    print(f"   Experiments: {', '.join(experiments)}")
    
    if args.skip_completed:
        print("   (Skipping completed experiments)")
    
    # Run experiments
    results = {}
    start_time = datetime.now()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Processing {exp}...")
        success = run_experiment(exp, skip_completed=args.skip_completed)
        results[exp] = "success" if success else "failed"
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600  # hours
    
    print("\n" + "="*80)
    print("📊 EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {sum(1 for r in results.values() if r == 'success')}")
    print(f"Failed: {sum(1 for r in results.values() if r == 'failed')}")
    print(f"Total time: {duration:.2f} hours")
    print("\nResults:")
    for exp, status in results.items():
        icon = "✅" if status == "success" else "❌"
        print(f"  {icon} {exp}: {status}")
    
    # Save summary
    summary_file = Path("outputs/experiments_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "timestamp": start_time.isoformat(),
        "duration_hours": duration,
        "results": results,
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print("\n💡 Tip: Run 'python scripts/run_experiments.py --compare' to compare results")


if __name__ == "__main__":
    main()
