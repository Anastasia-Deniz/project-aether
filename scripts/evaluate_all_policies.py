"""
Evaluate All Trained Policies - Simple Version
Finds all final_policy.pt files and evaluates them.

Usage:
    python scripts/evaluate_all_policies.py
    python scripts/evaluate_all_policies.py --num-samples 50
"""

import sys
import argparse
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_probe_compatibility(probe_path: str, training_config: dict, project_root: Path) -> bool:
    """
    Validate if a probe is compatible with the training parameters.

    Args:
        probe_path: Path to the probe pytorch directory
        training_config: Training configuration dict
        project_root: Path to project root

    Returns:
        True if probe appears compatible, False otherwise
    """
    try:
        # Get target parameters from training config
        env_config = training_config.get('env_config', {})
        target_model = env_config.get('model_id', 'CompVis/stable-diffusion-v1-4')
        target_steps = env_config.get('num_inference_steps', 20)

        # Get probe directory and timestamp
        probe_dir = Path(probe_path).parent.parent  # Go up from pytorch/ to probe run dir
        probe_timestamp = probe_dir.name.split('_')[1]  # Extract date from run_YYYYMMDD_HHMMSS

        # Look for compatible latents
        latents_dir = project_root / "data" / "latents"
        if not latents_dir.exists():
            return False

        for latent_run in latents_dir.glob("run_*"):
            latent_timestamp = latent_run.name.split('_')[1]
            # Check if from same day
            if latent_timestamp[:8] == probe_timestamp[:8]:
                metadata_file = latent_run / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            latent_model = metadata.get('model_id')
                            latent_steps = metadata.get('num_steps')

                            if (latent_model == target_model and
                                latent_steps == target_steps):
                                return True
                    except Exception:
                        pass

        return False
    except Exception:
        return False

from scripts.hyperparameter_search import (
    get_python_executable,
    compare_all_results,
)


def find_all_policies() -> List[Dict]:
    """Find all final_policy.pt files."""
    output_dir = Path("outputs/ppo")
    if not output_dir.exists():
        return []
    
    policies = []
    
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        final_policy = run_dir / "final_policy.pt"
        if not final_policy.exists():
            continue
        
        # Try to load training history to get info
        history_file = run_dir / "training_history.json"
        final_reward = None
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                    if history.get("rewards"):
                        final_reward = sum(history["rewards"][-10:]) / min(10, len(history["rewards"]))
            except:
                pass
        
        policies.append({
            "run_dir": str(run_dir),
            "policy_path": str(final_policy),
            "run_name": run_dir.name,
            "final_reward": final_reward,
            "timestamp": run_dir.stat().st_mtime,
        })
    
    # Sort by timestamp (newest first)
    policies.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return policies


def evaluate_policy(policy_path: Path, num_samples: int = 50) -> Dict:
    """Evaluate a single policy."""
    python_exe = get_python_executable()
    project_root = Path(__file__).parent.parent

    # Convert to absolute paths
    policy_path = policy_path.resolve() if not policy_path.is_absolute() else policy_path
    eval_script = (project_root / "scripts" / "evaluate_ppo.py").resolve()

    # Get probe path from training config first
    probe_path = None
    run_dir = policy_path.parent
    config_file = run_dir / "training_config.json"

    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                probe_path = config.get('probe_path')
                probe_run = config.get('probe_run')
                if probe_path:
                    # Verify the probe still exists and has probe_t19.pt
                    probe_t19_path = Path(probe_path) / "probe_t19.pt"
                    if probe_t19_path.exists():
                        # Additional validation: check if probe is compatible with training parameters
                        is_compatible = validate_probe_compatibility(
                            probe_path, config, project_root
                        )
                        if is_compatible:
                            print(f"Using probe from training config: {probe_run}")
                        else:
                            print(f"Warning: Probe from training config ({probe_run}) not compatible with training parameters, auto-detecting...")
                            probe_path = None
                    else:
                        print(f"Warning: Probe from training config ({probe_run}) missing probe_t19.pt, auto-detecting...")
                        probe_path = None
                else:
                    print("Warning: Training config exists but no probe_path found")
        except Exception as e:
            print(f"Warning: Could not load training config: {e}")

    # Auto-detect compatible probe with probe_t19.pt
    if probe_path is None:
        # Get target parameters from policy's training config or use defaults
        target_model = "CompVis/stable-diffusion-v1-4"  # Default
        target_steps = 20  # Default

        # Try to get parameters from policy's training config
        try:
            with open(run_dir / "training_config.json") as f:
                policy_config = json.load(f)
                env_config = policy_config.get('env_config', {})
                if env_config.get('model_id'):
                    target_model = env_config['model_id']
                if env_config.get('num_inference_steps'):
                    target_steps = env_config['num_inference_steps']
        except Exception:
            pass

        print(f"Looking for probes compatible with model: {target_model}, steps: {target_steps}")

        probe_dirs = sorted(
            (project_root / "checkpoints" / "probes").glob("run_*"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True
        )

        selected_probe = None
        selected_latents_run = None

        for probe_dir in probe_dirs:
            pytorch_dir = probe_dir / "pytorch"
            probe_t19 = pytorch_dir / "probe_t19.pt"

            if not probe_t19.exists():
                continue

            # Check if this probe was trained on compatible latents
            latents_dir = project_root / "data" / "latents"
            if latents_dir.exists():
                # Find latent runs that match the probe timestamp
                probe_timestamp = probe_dir.name.split('_')[1]  # Extract date from run_YYYYMMDD_HHMMSS
                compatible_latents = []

                for latent_run in latents_dir.glob("run_*"):
                    latent_timestamp = latent_run.name.split('_')[1]
                    # Check if timestamps are from the same day
                    if latent_timestamp[:8] == probe_timestamp[:8]:
                        metadata_file = latent_run / "metadata.json"
                        if metadata_file.exists():
                            try:
                                with open(metadata_file) as f:
                                    metadata = json.load(f)
                                    latent_model = metadata.get('model_id')
                                    latent_steps = metadata.get('num_steps')

                                    if (latent_model == target_model and
                                        latent_steps == target_steps):
                                        compatible_latents.append((latent_run, metadata))
                            except Exception:
                                pass

                if compatible_latents:
                    selected_probe = str(pytorch_dir)
                    selected_latents_run = compatible_latents[0][0].name
                    print(f"Using compatible probe: {probe_dir.name}")
                    print(f"  Latents: {selected_latents_run}")
                    print(f"  Model: {target_model}, Steps: {target_steps}")
                    break

        # Fallback: use most recent probe if no compatible one found
        if not selected_probe:
            print("No probe with compatible latents found, using most recent probe...")
            for probe_dir in probe_dirs:
                pytorch_dir = probe_dir / "pytorch"
                probe_t19 = pytorch_dir / "probe_t19.pt"
                if probe_t19.exists():
                    selected_probe = str(pytorch_dir)
                    print(f"Using fallback probe: {probe_dir.name} (most recent)")
                    break

        if selected_probe is None and probe_dirs:
            # Last fallback: use the most recent probe run even if it doesn't have probe_t19.pt
            selected_probe = str(probe_dirs[0] / "pytorch")
            print(f"Warning: No probe run with probe_t19.pt found, using latest: {probe_dirs[0].name}")

        probe_path = selected_probe

        if probe_path is None:
            print("Warning: No probe directories found")
    
    # Build command with absolute paths
    if " " in python_exe:
        cmd = python_exe.split() + [
            str(eval_script),
            "--policy_path", str(policy_path),
            "--num_samples", str(num_samples),
            "--seed", "42",  # Fixed seed for reproducibility
        ]
    else:
        cmd = [
            python_exe,
            str(eval_script),
            "--policy_path", str(policy_path),
            "--num_samples", str(num_samples),
            "--seed", "42",  # Fixed seed for reproducibility
        ]
    
    if probe_path:
        cmd.extend(["--probe_path", probe_path])
    
    try:
        # Ensure UTF-8 encoding for subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of failing
            env=env,
        )
        
        # Find the created evaluation directory
        eval_dir = project_root / "outputs/evaluation"
        if eval_dir.exists():
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
                        return metrics
        
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Evaluation failed with exit code {e.returncode}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Working directory: {project_root}")
        if e.stdout:
            # Show last 1000 chars of stdout
            stdout_lines = e.stdout.split('\n')
            print(f"  STDOUT (last 30 lines):")
            for line in stdout_lines[-30:]:
                print(f"    {line}")
        if e.stderr:
            # Show last 1000 chars of stderr
            stderr_lines = e.stderr.split('\n')
            print(f"  STDERR (last 30 lines):")
            for line in stderr_lines[-30:]:
                print(f"    {line}")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained policies")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples for evaluation (default: 50)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip policies that have already been evaluated",
    )
    
    args = parser.parse_args()
    
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
    print("📊 EVALUATING POLICIES")
    print(f"{'='*80}")
    print(f"Python: {get_python_executable()}")
    print(f"Evaluation samples: {args.num_samples}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for i, policy in enumerate(policies, 1):
        print(f"\n[{i}/{len(policies)}] Evaluating: {policy['run_name']}")
        print(f"  Policy: {policy['policy_path']}")
        
        # Check if already evaluated
        if args.skip_completed:
            eval_dir = Path("outputs/evaluation")
            if eval_dir.exists():
                eval_dirs = list(eval_dir.glob("eval_*"))
                for eval_dir_path in eval_dirs:
                    metrics_file = eval_dir_path / "evaluation_metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file) as f:
                                existing_metrics = json.load(f)
                                if existing_metrics.get("policy_path") == policy["policy_path"]:
                                    print(f"  ⏭️  Already evaluated, skipping")
                                    all_results.append(existing_metrics)
                                    continue
                        except:
                            pass
        
        # Evaluate
        metrics = evaluate_policy(Path(policy["policy_path"]), args.num_samples)
        
        if metrics:
            # Add policy info
            metrics["policy_path"] = policy["policy_path"]
            metrics["run_name"] = policy["run_name"]
            metrics["run_dir"] = policy["run_dir"]
            if policy["final_reward"]:
                metrics["training_reward"] = policy["final_reward"]
            
            # Add a unique identifier to prevent duplicate results
            metrics["evaluation_timestamp"] = datetime.now().isoformat()
            
            all_results.append(metrics)
            print(f"  ✅ SSR: {metrics.get('ssr', 0):.4f}, FPR: {metrics.get('fpr', 0):.4f}, LPIPS: {metrics.get('lpips_mean', 0):.4f}, Transport: {metrics.get('transport_cost_mean', 0):.2f}")
        else:
            print(f"  ❌ Evaluation failed")
    
    # Compare results
    if all_results:
        print(f"\n{'='*80}")
        print("🏆 COMPARING RESULTS")
        print(f"{'='*80}")
        compare_all_results(all_results)
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

