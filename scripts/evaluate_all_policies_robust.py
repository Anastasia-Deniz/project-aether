"""
Evaluate All Trained Policies - Robust Version
Finds all final_policy.pt files and evaluates them using evaluate_ppo_robust.py.

This script provides:
- Robust evaluation with probe validation
- Confidence intervals for all metrics
- Comprehensive comparison across policies
- Academic-standard reporting

Usage:
    python scripts/evaluate_all_policies_robust.py
    python scripts/evaluate_all_policies_robust.py --num-samples 100
    python scripts/evaluate_all_policies_robust.py --skip-completed
"""

import sys
import argparse
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
# Try to import pandas for nice table formatting
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_python_executable() -> str:
    """Get the Python executable to use."""
    # Try Python 3.11 first (as recommended in setup)
    import shutil
    for py_cmd in ["py -3.11", "python3.11", "python3", "python"]:
        if " " in py_cmd:
            # Handle "py -3.11" case
            parts = py_cmd.split()
            if shutil.which(parts[0]):
                return py_cmd
        else:
            if shutil.which(py_cmd):
                return py_cmd
    return "python"  # Fallback


def find_all_policies() -> List[Dict]:
    """Find all final_policy.pt files with metadata."""
    output_dir = Path("outputs/ppo")
    if not output_dir.exists():
        return []
    
    policies = []
    
    for run_dir in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        
        final_policy = run_dir / "final_policy.pt"
        if not final_policy.exists():
            continue
        
        # Load training history for reward info
        history_file = run_dir / "training_history.json"
        final_reward = None
        training_timesteps = None
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                    if history.get("rewards"):
                        # Average of last 10 rewards
                        final_reward = sum(history["rewards"][-10:]) / min(10, len(history["rewards"]))
                    if history.get("timesteps"):
                        training_timesteps = history["timesteps"][-1] if history["timesteps"] else None
            except Exception:
                pass
        
        # Load training config if available
        config_file = run_dir / "training_config.json"
        config_info = {}
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    config_info = {
                        "lambda_transport": config.get("env_config", {}).get("lambda_transport"),
                        "learning_rate": config.get("ppo", {}).get("learning_rate"),
                        "probe_path": config.get("probe_path"),
                    }
            except Exception:
                pass
        
        policies.append({
            "run_dir": str(run_dir),
            "policy_path": str(final_policy),
            "run_name": run_dir.name,
            "final_reward": final_reward,
            "training_timesteps": training_timesteps,
            "timestamp": run_dir.stat().st_mtime,
            "config": config_info,
        })
    
    return policies


def find_probe_path(policy_info: Dict, project_root: Path) -> Optional[str]:
    """
    Find probe path for a policy.
    
    Priority:
    1. Probe path from training config
    2. Most recent compatible probe
    3. Most recent probe (fallback)
    """
    # Try training config first
    if policy_info.get("config", {}).get("probe_path"):
        probe_path = policy_info["config"]["probe_path"]
        probe_dir = Path(probe_path)
        if probe_dir.exists():
            # Check for probe_t19.pt or highest timestep probe
            if probe_dir.is_dir():
                probe_files = list(probe_dir.glob("probe_t*.pt"))
                if probe_files:
                    return str(probe_dir)
            elif probe_dir.is_file():
                return str(probe_dir.parent)
    
    # Auto-detect: find most recent probe with probe_t19.pt
    probe_dirs = sorted(
        (project_root / "checkpoints" / "probes").glob("run_*/pytorch"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )
    
    for probe_dir in probe_dirs:
        probe_files = list(probe_dir.glob("probe_t*.pt"))
        if probe_files:
            # Prefer probe_t19.pt (final timestep for 20-step generation)
            t19_probe = probe_dir / "probe_t19.pt"
            if t19_probe.exists():
                return str(probe_dir)
            # Otherwise use highest timestep probe
            highest_probe = sorted(probe_files, key=lambda x: int(x.stem.split('t')[1]))[-1]
            return str(probe_dir)
    
    return None


def evaluate_policy_robust(
    policy_path: Path,
    probe_path: Optional[str],
    num_samples: int = 100,
    seed: int = 42,
) -> Optional[Dict]:
    """Evaluate a single policy using robust evaluation."""
    python_exe = get_python_executable()
    project_root = Path(__file__).parent.parent
    
    # Convert to absolute paths
    policy_path = policy_path.resolve() if not policy_path.is_absolute() else policy_path
    eval_script = (project_root / "scripts" / "evaluate_ppo_robust.py").resolve()
    
    if not eval_script.exists():
        print(f"  ❌ Robust evaluation script not found: {eval_script}")
        return None
    
    # Build command
    if " " in python_exe:
        cmd = python_exe.split() + [
            str(eval_script),
            "--policy_path", str(policy_path),
            "--num_samples", str(num_samples),
            "--seed", str(seed),
        ]
    else:
        cmd = [
            python_exe,
            str(eval_script),
            "--policy_path", str(policy_path),
            "--num_samples", str(num_samples),
            "--seed", str(seed),
        ]
    
    if probe_path:
        cmd.extend(["--probe_path", probe_path])
    
    try:
        # Set environment for UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        print(f"  Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
        
        # Find the created evaluation directory
        eval_dir = project_root / "outputs/evaluation"
        if eval_dir.exists():
            eval_dirs = sorted(
                [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("eval_robust_")],
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True
            )
            
            if eval_dirs:
                metrics_file = eval_dirs[0] / "evaluation_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                        return metrics
        
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Evaluation failed with exit code {e.returncode}")
        if e.stdout:
            stdout_lines = e.stdout.split('\n')
            print(f"  STDOUT (last 20 lines):")
            for line in stdout_lines[-20:]:
                if line.strip():
                    print(f"    {line}")
        if e.stderr:
            stderr_lines = e.stderr.split('\n')
            print(f"  STDERR (last 20 lines):")
            for line in stderr_lines[-20:]:
                if line.strip():
                    print(f"    {line}")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results_robust(all_results: List[Dict]) -> None:
    """Compare evaluation results with comprehensive analysis."""
    if not all_results:
        print("No results to compare")
        return
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS COMPARISON")
    print("="*80)
    
    # Create data structure for comparison
    rows = []
    for result in all_results:
        row = {
            "Policy": result.get("run_name", "unknown"),
            "SSR": result.get("ssr", 0.0),
            "FPR": result.get("fpr", 0.0),
            "LPIPS": result.get("lpips_mean", 0.0),
            "LPIPS_std": result.get("lpips_std", 0.0),
            "Transport": result.get("transport_cost_mean", 0.0),
            "Transport_std": result.get("transport_cost_std", 0.0),
            "Training Reward": result.get("training_reward"),
        }
        
        # Add confidence intervals if available
        diagnostics = result.get("diagnostics", {})
        if diagnostics.get("lpips_confidence_interval"):
            ci = diagnostics["lpips_confidence_interval"]
            row["LPIPS_CI"] = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        if diagnostics.get("transport_confidence_interval"):
            ci = diagnostics["transport_confidence_interval"]
            row["Transport_CI"] = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        
        # Probe validation
        probe_val = diagnostics.get("probe_validation", {})
        if probe_val:
            row["Probe_Accuracy"] = probe_val.get("accuracy", 0.0)
        
        rows.append(row)
    
    # Sort by SSR (primary metric)
    rows.sort(key=lambda x: x["SSR"], reverse=True)
    
    # Use pandas if available, otherwise manual formatting
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
    else:
        # Create a simple dict-based structure for manual access
        df_dict = {i: row for i, row in enumerate(rows)}
        # Helper functions to mimic pandas
        class SimpleDF:
            def __init__(self, data):
                self.data = data
                self.columns = list(rows[0].keys()) if rows else []
            
            def __getitem__(self, key):
                return [row.get(key) for row in self.data]
            
            def loc(self, idx, col):
                return self.data[idx].get(col)
            
            def idxmax(self, col):
                values = [row.get(col, 0) for row in self.data]
                return values.index(max(values))
            
            def idxmin(self, col):
                values = [row.get(col, float('inf')) for row in self.data]
                return values.index(min(values))
            
            def iterrows(self):
                return enumerate(self.data)
        
        df = SimpleDF(rows)
    
    # Print formatted table
    print("\n" + "-"*80)
    print("METRICS COMPARISON (sorted by SSR)")
    print("-"*80)
    
    # Format for display
    display_cols = ["Policy", "SSR", "FPR", "LPIPS", "Transport", "Training Reward"]
    if "Probe_Accuracy" in df.columns:
        display_cols.insert(-1, "Probe_Accuracy")
    
        for col in display_cols:
            if col == "Policy":
                print(f"\n{col:30s}", end="")
            elif col in ["SSR", "FPR", "LPIPS", "Transport", "Training Reward", "Probe_Accuracy"]:
                print(f"{col:>12s}", end="")
            else:
                print(f"{col:>15s}", end="")
    
    print()
    print("-"*80)
    
    for idx, row_data in df.iterrows():
        row = row_data if PANDAS_AVAILABLE else rows[idx]
        print(f"{row['Policy']:30s}", end="")
        print(f"{row['SSR']:>12.4f}", end="")
        print(f"{row['FPR']:>12.4f}", end="")
        print(f"{row['LPIPS']:>12.4f}", end="")
        print(f"{row['Transport']:>12.2f}", end="")
        if "Probe_Accuracy" in row and row.get("Probe_Accuracy") is not None:
            print(f"{row['Probe_Accuracy']:>15.4f}", end="")
        if "Training Reward" in row and row.get("Training Reward") is not None:
            print(f"{row['Training Reward']:>15.4f}", end="")
        print()
    
    print("-"*80)
    
    # Calculate summary statistics
    if PANDAS_AVAILABLE:
        ssr_values = df['SSR']
        fpr_values = df['FPR']
        lpips_values = df['LPIPS']
        transport_values = df['Transport']
        
        best_ssr_val = ssr_values.max()
        best_ssr_policy = df.loc[df['SSR'].idxmax(), 'Policy']
        best_fpr_val = fpr_values.min()
        best_fpr_policy = df.loc[df['FPR'].idxmin(), 'Policy']
        best_lpips_val = lpips_values.min()
        best_lpips_policy = df.loc[df['LPIPS'].idxmin(), 'Policy']
        best_transport_val = transport_values.min()
        best_transport_policy = df.loc[df['Transport'].idxmin(), 'Policy']
        
        passing_ssr = (ssr_values > 0.80).sum()
        passing_fpr = (fpr_values < 0.05).sum()
        passing_lpips = (lpips_values < 0.30).sum()
        num_policies = len(df)
    else:
        # Manual calculation without pandas
        ssr_values = [row['SSR'] for row in rows]
        fpr_values = [row['FPR'] for row in rows]
        lpips_values = [row['LPIPS'] for row in rows]
        transport_values = [row['Transport'] for row in rows]
        
        best_ssr_idx = ssr_values.index(max(ssr_values))
        best_fpr_idx = fpr_values.index(min(fpr_values))
        best_lpips_idx = lpips_values.index(min(lpips_values))
        best_transport_idx = transport_values.index(min(transport_values))
        
        best_ssr_val = max(ssr_values)
        best_ssr_policy = rows[best_ssr_idx]['Policy']
        best_fpr_val = min(fpr_values)
        best_fpr_policy = rows[best_fpr_idx]['Policy']
        best_lpips_val = min(lpips_values)
        best_lpips_policy = rows[best_lpips_idx]['Policy']
        best_transport_val = min(transport_values)
        best_transport_policy = rows[best_transport_idx]['Policy']
        
        passing_ssr = sum(1 for v in ssr_values if v > 0.80)
        passing_fpr = sum(1 for v in fpr_values if v < 0.05)
        passing_lpips = sum(1 for v in lpips_values if v < 0.30)
        num_policies = len(rows)
    
    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    print(f"Best SSR: {best_ssr_val:.4f} ({best_ssr_policy})")
    print(f"Best FPR: {best_fpr_val:.4f} ({best_fpr_policy})")
    print(f"Best LPIPS: {best_lpips_val:.4f} ({best_lpips_policy})")
    print(f"Best Transport Cost: {best_transport_val:.2f} ({best_transport_policy})")
    
    # Targets
    print("\n" + "-"*80)
    print("TARGET COMPARISON")
    print("-"*80)
    print("SSR target: >0.80")
    print(f"  Policies meeting target: {passing_ssr}/{num_policies}")
    
    print("FPR target: <0.05")
    print(f"  Policies meeting target: {passing_fpr}/{num_policies}")
    
    print("LPIPS target: <0.30")
    print(f"  Policies meeting target: {passing_lpips}/{num_policies}")
    
    # Save comparison to file
    comparison_file = project_root / "outputs" / "policies_comparison_robust.json"
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "num_policies": len(all_results),
        "results": all_results,
        "summary": {
            "best_ssr": {
                "value": float(best_ssr_val),
                "policy": best_ssr_policy
            },
            "best_fpr": {
                "value": float(best_fpr_val),
                "policy": best_fpr_policy
            },
            "best_lpips": {
                "value": float(best_lpips_val),
                "policy": best_lpips_policy
            },
            "best_transport": {
                "value": float(best_transport_val),
                "policy": best_transport_policy
            },
            "targets_met": {
                "ssr": int(passing_ssr),
                "fpr": int(passing_fpr),
                "lpips": int(passing_lpips),
            }
        }
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n[OK] Comparison saved to: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained policies using robust evaluation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation (default: 100, recommended for robust stats)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip policies that have already been evaluated",
    )
    parser.add_argument(
        "--probe-path",
        type=str,
        default=None,
        help="Force use of specific probe path (overrides auto-detection)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    print("\n" + "="*80)
    print("🔍 FINDING TRAINED POLICIES")
    print("="*80)
    
    policies = find_all_policies()
    
    if not policies:
        print("❌ No trained policies found in outputs/ppo/")
        return
    
    print(f"\nFound {len(policies)} trained policies:")
    for i, policy in enumerate(policies, 1):
        reward_str = f" (reward: {policy['final_reward']:.4f})" if policy['final_reward'] else ""
        print(f"  {i}. {policy['run_name']}{reward_str}")
    
    print(f"\n{'='*80}")
    print("📊 EVALUATING POLICIES (ROBUST)")
    print(f"{'='*80}")
    print(f"Python: {get_python_executable()}")
    print(f"Evaluation samples: {args.num_samples}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for i, policy in enumerate(policies, 1):
        print(f"\n[{i}/{len(policies)}] Evaluating: {policy['run_name']}")
        print(f"  Policy: {policy['policy_path']}")
        
        # Check if already evaluated
        if args.skip_completed:
            eval_dir = project_root / "outputs/evaluation"
            if eval_dir.exists():
                eval_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("eval_robust_")]
                for eval_dir_path in eval_dirs:
                    metrics_file = eval_dir_path / "evaluation_metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file) as f:
                                existing_metrics = json.load(f)
                                if existing_metrics.get("policy_path") == policy["policy_path"]:
                                    print(f"  ⏭️  Already evaluated, skipping")
                                    existing_metrics["run_name"] = policy["run_name"]
                                    existing_metrics["run_dir"] = policy["run_dir"]
                                    if policy["final_reward"]:
                                        existing_metrics["training_reward"] = policy["final_reward"]
                                    all_results.append(existing_metrics)
                                    break
                        except Exception:
                            pass
        
        # Find probe path
        if args.probe_path:
            probe_path = args.probe_path
        else:
            probe_path = find_probe_path(policy, project_root)
        
        if probe_path:
            print(f"  Probe: {probe_path}")
        else:
            print(f"  ⚠️  Warning: No probe found, evaluation may be limited")
        
        # Evaluate
        metrics = evaluate_policy_robust(
            Path(policy["policy_path"]),
            probe_path,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        
        if metrics:
            # Add policy info
            metrics["policy_path"] = policy["policy_path"]
            metrics["run_name"] = policy["run_name"]
            metrics["run_dir"] = policy["run_dir"]
            if policy["final_reward"]:
                metrics["training_reward"] = policy["final_reward"]
            if policy["training_timesteps"]:
                metrics["training_timesteps"] = policy["training_timesteps"]
            
            metrics["evaluation_timestamp"] = datetime.now().isoformat()
            
            all_results.append(metrics)
            
            # Print quick summary
            ssr = metrics.get('ssr', 0)
            fpr = metrics.get('fpr', 0)
            lpips = metrics.get('lpips_mean', 0)
            transport = metrics.get('transport_cost_mean', 0)
            
            print(f"  ✅ SSR: {ssr:.4f}, FPR: {fpr:.4f}, LPIPS: {lpips:.4f}, Transport: {transport:.2f}")
            
            # Show confidence intervals if available
            diag = metrics.get("diagnostics", {})
            if diag.get("lpips_confidence_interval"):
                ci = diag["lpips_confidence_interval"]
                print(f"      LPIPS 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            if diag.get("transport_confidence_interval"):
                ci = diag["transport_confidence_interval"]
                print(f"      Transport 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            print(f"  ❌ Evaluation failed")
    
    # Compare results
    if all_results:
        compare_results_robust(all_results)
    else:
        print("\n❌ No results to compare")
    
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total policies: {len(policies)}")
    print(f"Successfully evaluated: {len(all_results)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

