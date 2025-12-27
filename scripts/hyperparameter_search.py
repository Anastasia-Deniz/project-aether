"""
Comprehensive Hyperparameter Search Script
Runs training + evaluation for all experiments and finds the best configuration.

This script:
1. Runs training for each experiment config
2. Evaluates each trained policy
3. Compares all results (SSR, FPR, LPIPS, Transport Cost)
4. Identifies the best configuration
5. Saves comprehensive comparison report

Usage:
    python scripts/hyperparameter_search.py
    python scripts/hyperparameter_search.py --skip-completed
    python scripts/hyperparameter_search.py --experiments exp_lambda_0.5 exp_lambda_0.8
    python scripts/hyperparameter_search.py --quick  # Fast test with fewer samples
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import platform

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Global variable to store Python executable (can be overridden by --python arg)
_python_executable = None

def get_python_executable() -> str:
    """
    Get the Python executable to use.
    On Windows, try to use 'py -3.11' if available, otherwise use sys.executable.
    """
    # Use override if set
    if _python_executable:
        return _python_executable
    
    if platform.system() == "Windows":
        # Try to use py -3.11 launcher (as in setup_env.bat)
        try:
            result = subprocess.run(
                ["py", "-3.11", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "py -3.11"
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
    
    # Fallback to current Python
    return sys.executable


def get_experiment_configs() -> List[str]:
    """Get all experiment config files."""
    exp_dir = Path("configs/experiments")
    if not exp_dir.exists():
        return []
    
    configs = sorted(exp_dir.glob("*.yaml"))
    return [c.stem for c in configs]


def load_experiment_config(config_name: str) -> dict:
    """Load experiment config."""
    config_path = f"configs/experiments/{config_name}.yaml"
    if not Path(config_path).exists():
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def check_training_completed(experiment_name: str) -> Optional[Path]:
    """Check if training completed and return policy path."""
    output_dir = Path("outputs/ppo")
    if not output_dir.exists():
        return None
    
    # Look for experiment directory (try both patterns)
    # Pattern 1: exp_lambda_0.5_20251227_022301
    exp_dirs = list(output_dir.glob(f"{experiment_name}_*"))
    
    # Pattern 2: aether_ppo_20251227_022301 (default name, check by timestamp or config)
    if not exp_dirs:
        # Try to find by checking all directories and matching by config
        all_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        for d in all_dirs:
            final_policy = d / "final_policy.pt"
            if final_policy.exists():
                # Check if this might be our experiment by checking training_history
                # or by being the most recent if we can't match
                exp_dirs.append(d)
    
    if not exp_dirs:
        return None
    
    # Get latest run
    latest_dir = max(exp_dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    final_policy = latest_dir / "final_policy.pt"
    
    if final_policy.exists():
        return final_policy
    
    return None


def check_evaluation_completed(experiment_name: str) -> bool:
    """Check if evaluation already completed."""
    eval_dir = Path("outputs/evaluation")
    if not eval_dir.exists():
        return False
    
    # Look for evaluation with this experiment name
    eval_dirs = list(eval_dir.glob("eval_*"))
    for eval_dir_path in eval_dirs:
        metrics_file = eval_dir_path / "evaluation_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    # Check if this evaluation matches the experiment
                    # (we'll store experiment name in the metrics)
                    if metrics.get("experiment_name") == experiment_name:
                        return True
            except:
                pass
    
    return False


def run_training(config_name: str, skip_completed: bool = False) -> Optional[Path]:
    """Run training for an experiment."""
    config_path = f"configs/experiments/{config_name}.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        return None
    
    # Get experiment name from config
    config = load_experiment_config(config_name)
    exp_name = config.get("training", {}).get("experiment_name", config_name)
    
    # Check if already completed
    if skip_completed:
        policy_path = check_training_completed(exp_name)
        if policy_path:
            print(f"⏭️  Training already completed: {policy_path}")
            return policy_path
    
    print("\n" + "="*80)
    print(f"🚀 TRAINING: {config_name}")
    print("="*80)
    
    # Show key parameters
    if config:
        print("\nKey Parameters:")
        if "ppo" in config:
            ppo = config["ppo"]
            print(f"  Total timesteps: {ppo.get('total_timesteps', 'N/A'):,}")
            print(f"  Learning rate: {ppo.get('learning_rate', 'N/A')}")
            print(f"  Batch size: {ppo.get('batch_size', 'N/A')}")
            print(f"  Epochs: {ppo.get('n_epochs', 'N/A')}")
        if "reward" in config:
            print(f"  Lambda transport: {config['reward'].get('lambda_transport', 'N/A')}")
        if "env" in config:
            print(f"  Intervention: [{config['env'].get('intervention_start', 'N/A')}, {config['env'].get('intervention_end', 'N/A')}]")
    
    print("="*80)
    
    # Run training
    python_exe = get_python_executable()
    if " " in python_exe:
        # Handle "py -3.11" style commands
        cmd = python_exe.split() + ["scripts/train_ppo.py", "--config", config_path]
    else:
        cmd = [python_exe, "scripts/train_ppo.py", "--config", config_path]
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent.parent,
            capture_output=False,
        )
        duration = time.time() - start_time
        
        # Find the created policy - look in the most recent directory
        # The training script creates directories with timestamp, so find the newest one
        output_dir = Path("outputs/ppo")
        if output_dir.exists():
            # Get all directories, sort by modification time
            all_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            if all_dirs:
                # Get the most recently modified directory
                latest_dir = max(all_dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0)
                final_policy = latest_dir / "final_policy.pt"
                
                if final_policy.exists():
                    print(f"\n✅ Training completed in {duration/60:.1f} minutes")
                    print(f"   Policy saved: {final_policy}")
                    return final_policy
        
        # Fallback: try the original method
        policy_path = check_training_completed(exp_name)
        if policy_path:
            print(f"\n✅ Training completed in {duration/60:.1f} minutes")
            print(f"   Policy saved: {policy_path}")
            return policy_path
        else:
            print(f"\n⚠️  Training completed but policy not found")
            print(f"   Check outputs/ppo/ for the most recent directory")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return None
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return None


def run_evaluation(
    config_name: str,
    policy_path: Path,
    num_samples: int = 50,
    skip_completed: bool = False
) -> Optional[Dict]:
    """Run evaluation for a trained policy."""
    config = load_experiment_config(config_name)
    exp_name = config.get("training", {}).get("experiment_name", config_name)
    
    # Check if already evaluated
    if skip_completed and check_evaluation_completed(exp_name):
        print(f"⏭️  Evaluation already completed for {config_name}")
        # Try to load existing results
        eval_dir = Path("outputs/evaluation")
        if eval_dir.exists():
            eval_dirs = list(eval_dir.glob("eval_*"))
            for eval_dir_path in sorted(eval_dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
                metrics_file = eval_dir_path / "evaluation_metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                            if metrics.get("experiment_name") == exp_name:
                                print(f"   Loaded existing results from {eval_dir_path}")
                                return metrics
                    except Exception as e:
                        print(f"   Warning: Could not load existing results: {e}")
                        pass
    
    print("\n" + "="*80)
    print(f"📊 EVALUATING: {config_name}")
    print("="*80)
    print(f"   Policy: {policy_path}")
    print(f"   Samples: {num_samples}")
    print("="*80)
    
    # Get probe path (auto-detect latest)
    probe_path = config.get("reward", {}).get("probe_path", "auto")
    if probe_path == "auto" or not Path(probe_path).exists():
        probe_dirs = sorted(
            Path("checkpoints/probes").glob("run_*"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True
        )
        if probe_dirs:
            probe_path = str(probe_dirs[0] / "pytorch")
        else:
            probe_path = None
    
    # Get environment settings from config
    env_config = config.get("env", {})
    
    # Run evaluation
    python_exe = get_python_executable()
    if " " in python_exe:
        # Handle "py -3.11" style commands
        cmd = python_exe.split() + [
            "scripts/evaluate_ppo.py",
            "--policy_path", str(policy_path),
            "--num_samples", str(num_samples),
        ]
    else:
        cmd = [
            python_exe,
            "scripts/evaluate_ppo.py",
            "--policy_path", str(policy_path),
            "--num_samples", str(num_samples),
        ]
    
    if probe_path:
        cmd.extend(["--probe_path", probe_path])
    
    if "num_inference_steps" in env_config:
        cmd.extend(["--num_inference_steps", str(env_config["num_inference_steps"])])
    
    if "intervention_start" in env_config:
        cmd.extend(["--intervention_start", str(env_config["intervention_start"])])
    
    if "intervention_end" in env_config:
        cmd.extend(["--intervention_end", str(env_config["intervention_end"])])
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        duration = time.time() - start_time
        
        # Find the created evaluation directory
        eval_dir = Path("outputs/evaluation")
        if not eval_dir.exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
        
        eval_dirs = sorted(
            list(eval_dir.glob("eval_*")),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True
        )
        
        if eval_dirs:
            metrics_file = eval_dirs[0] / "evaluation_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    metrics["experiment_name"] = exp_name
                    metrics["config_name"] = config_name
                    metrics["policy_path"] = str(policy_path)
                    metrics["evaluation_dir"] = str(eval_dirs[0])
                    metrics["evaluation_time_seconds"] = duration
                    
                    # Save updated metrics
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    print(f"\n✅ Evaluation completed in {duration/60:.1f} minutes")
                    print(f"   SSR: {metrics.get('ssr', 0):.4f}")
                    print(f"   FPR: {metrics.get('fpr', 0):.4f}")
                    print(f"   LPIPS: {metrics.get('lpips_mean', 0):.4f}")
                    print(f"   Transport Cost: {metrics.get('transport_cost_mean', 0):.4f}")
                    
                    return metrics
        
        print(f"\n⚠️  Evaluation completed but metrics not found")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-1000:])  # Last 1000 chars
        if e.stderr:
            print("STDERR:", e.stderr[-1000:])
        return None
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted by user")
        return None


def compute_score(metrics: Dict) -> float:
    """
    Compute a composite score for ranking experiments.
    
    Higher is better. Formula:
    score = SSR * 0.5 - FPR * 0.3 - (LPIPS/0.3) * 0.1 - (TransportCost/100) * 0.1
    
    This prioritizes SSR (safety success) while penalizing FPR, LPIPS, and transport cost.
    """
    ssr = metrics.get("ssr", 0.0)
    fpr = metrics.get("fpr", 1.0)  # Default to worst case
    lpips = metrics.get("lpips_mean", 1.0)  # Default to worst case
    transport = metrics.get("transport_cost_mean", 1000.0)  # Default to worst case
    
    # Normalize and weight
    score = (
        ssr * 0.5 +  # SSR is most important (0-1, higher better)
        (1.0 - min(fpr, 1.0)) * 0.3 +  # FPR penalty (0-1, lower better, so invert)
        (1.0 - min(lpips / 0.3, 1.0)) * 0.1 +  # LPIPS penalty (normalize to 0.3)
        (1.0 - min(transport / 100.0, 1.0)) * 0.1  # Transport penalty (normalize to 100)
    )
    
    return score


def compare_all_results(results: List[Dict]) -> None:
    """Compare and rank all experiment results."""
    if not results:
        print("❌ No results to compare")
        return
    
    # Compute scores and sort
    for r in results:
        r["composite_score"] = compute_score(r)
    
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    
    print("\n" + "="*80)
    print("🏆 HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    
    # Print comparison table
    print(f"\n{'Rank':<6} {'Experiment':<30} {'SSR':<8} {'FPR':<8} {'LPIPS':<10} {'Transport':<12} {'Score':<8}")
    print("-" * 110)
    
    for i, r in enumerate(results, 1):
        exp_name = r.get("config_name", r.get("experiment_name", "unknown"))
        ssr = r.get("ssr", 0.0)
        fpr = r.get("fpr", 0.0)
        lpips = r.get("lpips_mean", 0.0)
        transport = r.get("transport_cost_mean", 0.0)
        score = r.get("composite_score", 0.0)
        
        # Highlight best
        marker = "🏆" if i == 1 else "  "
        
        print(f"{marker} {i:<4} {exp_name:<30} {ssr:<8.4f} {fpr:<8.4f} {lpips:<10.4f} {transport:<12.2f} {score:<8.4f}")
    
    # Print best experiment details
    if results:
        best = results[0]
        print("\n" + "="*80)
        print("🏆 BEST CONFIGURATION")
        print("="*80)
        print(f"Experiment: {best.get('config_name', 'unknown')}")
        print(f"Composite Score: {best.get('composite_score', 0):.4f}")
        print(f"\nMetrics:")
        print(f"  SSR (Safety Success Rate): {best.get('ssr', 0):.4f} {'✅' if best.get('ssr', 0) > 0.80 else '❌'} (target: >0.80)")
        print(f"  FPR (False Positive Rate): {best.get('fpr', 0):.4f} {'✅' if best.get('fpr', 0) < 0.05 else '❌'} (target: <0.05)")
        print(f"  LPIPS (Perceptual Distance): {best.get('lpips_mean', 0):.4f} {'✅' if best.get('lpips_mean', 0) < 0.30 else '❌'} (target: <0.30)")
        print(f"  Transport Cost: {best.get('transport_cost_mean', 0):.4f} (minimize)")
        print(f"\nPolicy Path: {best.get('policy_path', 'N/A')}")
        print(f"Evaluation Dir: {best.get('evaluation_dir', 'N/A')}")
        
        # Show config parameters
        config_name = best.get("config_name")
        if config_name:
            config = load_experiment_config(config_name)
            if config:
                print(f"\nConfiguration Parameters:")
                if "ppo" in config:
                    ppo = config["ppo"]
                    print(f"  Total timesteps: {ppo.get('total_timesteps', 'N/A'):,}")
                    print(f"  Learning rate: {ppo.get('learning_rate', 'N/A')}")
                    print(f"  Batch size: {ppo.get('batch_size', 'N/A')}")
                    print(f"  Epochs: {ppo.get('n_epochs', 'N/A')}")
                if "reward" in config:
                    print(f"  Lambda transport: {config['reward'].get('lambda_transport', 'N/A')}")
                if "env" in config:
                    env = config["env"]
                    print(f"  Intervention: [{env.get('intervention_start', 'N/A')}, {env.get('intervention_end', 'N/A')}]")
    
    # Save comparison
    comparison_file = Path("outputs/hyperparameter_search_results.json")
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "best_experiment": results[0] if results else None,
        "all_results": results,
    }
    
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Full results saved to: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive hyperparameter search with training + evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (training + evaluation)
  python scripts/hyperparameter_search.py
  
  # Use Python 3.11 (as set up by setup_env.bat)
  python scripts/hyperparameter_search.py --python "py -3.11"
  
  # Skip already completed experiments
  python scripts/hyperparameter_search.py --skip-completed
  
  # Run specific experiments
  python scripts/hyperparameter_search.py --experiments exp_lambda_0.5 exp_lambda_0.8
  
  # Quick test with fewer evaluation samples
  python scripts/hyperparameter_search.py --quick
  
  # Only evaluate (skip training)
  python scripts/hyperparameter_search.py --evaluate-only
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
        help="Skip experiments that have already completed training and evaluation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: fewer evaluation samples (10 instead of 50)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation (skip training, assumes policies exist)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples for evaluation (default: 50)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable to use (default: auto-detect, tries 'py -3.11' on Windows)",
    )
    
    args = parser.parse_args()
    
    # Set Python executable if specified
    if args.python:
        global _python_executable
        _python_executable = args.python
    
    # Get experiments to run
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = get_experiment_configs()
    
    if not experiments:
        print("❌ No experiments found in configs/experiments/")
        print("   Create experiment configs first!")
        return
    
    # Adjust num_samples for quick mode
    num_samples = 10 if args.quick else args.num_samples
    
    # Get Python executable info
    python_exe = get_python_executable()
    
    print("\n" + "="*80)
    print("🔬 HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Python: {python_exe}")
    print(f"Experiments: {len(experiments)}")
    print(f"Evaluation samples: {num_samples}")
    print(f"Skip completed: {args.skip_completed}")
    print(f"Evaluate only: {args.evaluate_only}")
    print("="*80)
    
    # Run experiments
    all_results = []
    start_time = datetime.now()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(experiments)}] Processing: {exp}")
        print(f"{'='*80}")
        
        # Run training (unless evaluate-only)
        policy_path = None
        if not args.evaluate_only:
            policy_path = run_training(exp, skip_completed=args.skip_completed)
            if not policy_path:
                print(f"❌ Skipping evaluation for {exp} (training failed or skipped)")
                continue
        else:
            # Find existing policy
            config = load_experiment_config(exp)
            exp_name = config.get("training", {}).get("experiment_name", exp)
            policy_path = check_training_completed(exp_name)
            if not policy_path:
                print(f"❌ No policy found for {exp}, skipping")
                continue
        
        # Run evaluation
        metrics = run_evaluation(
            exp,
            policy_path,
            num_samples=num_samples,
            skip_completed=args.skip_completed
        )
        
        if metrics:
            all_results.append(metrics)
        else:
            print(f"⚠️  No metrics for {exp}")
    
    # Compare results
    if all_results:
        compare_all_results(all_results)
    else:
        print("\n❌ No results to compare")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600  # hours
    
    print("\n" + "="*80)
    print("📊 SEARCH SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful evaluations: {len(all_results)}")
    print(f"Total time: {duration:.2f} hours")
    print("="*80)
    
    if all_results:
        print(f"\n🏆 Best experiment: {all_results[0].get('config_name', 'unknown')}")
        print(f"   SSR: {all_results[0].get('ssr', 0):.4f}")
        print(f"   FPR: {all_results[0].get('fpr', 0):.4f}")


if __name__ == "__main__":
    main()

