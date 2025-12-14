"""
Run multiple PPO training experiments with different hyperparameters.

This script runs all experiments sequentially and saves results for comparison.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --experiments exp_lambda_0.3 exp_lambda_0.5
    python scripts/run_experiments.py --skip-completed
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_experiment_configs():
    """Get all experiment config files."""
    exp_dir = Path("configs/experiments")
    if not exp_dir.exists():
        return []
    
    configs = sorted(exp_dir.glob("*.yaml"))
    return [c.stem for c in configs]

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

def run_experiment(config_name: str, skip_completed: bool = False) -> bool:
    """Run a single experiment."""
    config_path = f"configs/experiments/{config_name}.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    # Check if already completed
    if skip_completed:
        # Extract experiment name from config
        with open(config_path) as f:
            for line in f:
                if "experiment_name:" in line:
                    exp_name = line.split(":")[1].strip().strip('"').strip("'")
                    if check_experiment_completed(exp_name):
                        print(f"⏭️  Skipping {config_name} (already completed)")
                        return True
                    break
    
    print("\n" + "="*80)
    print(f"🚀 Running Experiment: {config_name}")
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
        
        print(f"✅ Experiment {config_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {config_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Experiment {config_name} interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run PPO training experiments")
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
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        configs = get_experiment_configs()
        print("Available experiments:")
        for config in configs:
            status = "✅" if check_experiment_completed(config) else "⏳"
            print(f"  {status} {config}")
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

if __name__ == "__main__":
    main()

