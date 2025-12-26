"""
Project Aether - Complete Experiment Runner
Runs the full pipeline from setup verification to evaluation.

This script:
1. Verifies setup
2. (Optional) Runs Phase 1 if needed
3. Trains PPO policy (Phase 2)
4. Evaluates policy (Phase 3)

Usage:
    python run_experiments_complete.py --skip_phase1  # Use existing probes
    python run_experiments_complete.py --full         # Run everything from scratch
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_latest_probe():
    """Find the most recent probe checkpoint."""
    probe_dir = PROJECT_ROOT / "checkpoints" / "probes"
    if not probe_dir.exists():
        return None
    
    runs = sorted(probe_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return None
    
    latest = runs[0]
    probe_path = latest / "pytorch"
    
    if probe_path.exists() and list(probe_path.glob("probe_t*.pt")):
        return str(probe_path)
    
    return None


def run_setup_test():
    """Run setup verification."""
    print("\n" + "="*60)
    print("STEP 1: VERIFYING SETUP")
    print("="*60)
    
    # Use py -3.11 to match setup_env.bat
    import shutil
    python_cmd = shutil.which("py")
    if python_cmd:
        cmd = [python_cmd, "-3.11", "scripts/test_setup.py"]
    else:
        cmd = [sys.executable, "scripts/test_setup.py"]
    
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT)
    )
    
    if result.returncode != 0:
        print("\n⚠ Setup test failed. Please fix issues before proceeding.")
        return False
    
    print("\n✓ Setup verified successfully!")
    return True


def run_phase1(num_samples=50, num_steps=20):
    """Run Phase 1: Collect latents and train probes."""
    print("\n" + "="*60)
    print("STEP 2: PHASE 1 - COLLECTING LATENTS & TRAINING PROBES")
    print("="*60)
    
    # Step 2.1: Collect latents
    print("\n--- Collecting Latents ---")
    import shutil
    python_cmd = shutil.which("py")
    if python_cmd:
        python_exec = [python_cmd, "-3.11"]
    else:
        python_exec = [sys.executable]
    
    collect_cmd = python_exec + [
        "scripts/collect_latents.py",
        "--num_samples", str(num_samples),
        "--num_steps", str(num_steps),
        "--device", "cuda",
        "--focus_nudity",
        "--hard_only",
        "--min_nudity_pct", "50.0",
        "--min_inappropriate_pct", "60.0",
    ]
    
    result = subprocess.run(collect_cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("✗ Latent collection failed!")
        return None
    
    # Find the latest latents directory
    latents_dir = PROJECT_ROOT / "data" / "latents"
    runs = sorted(latents_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("✗ No latents directory found!")
        return None
    
    latents_path = runs[0]
    print(f"✓ Latents collected: {latents_path}")
    
    # Step 2.2: Train probes
    print("\n--- Training Probes ---")
    train_cmd = python_exec + [
        "scripts/train_probes.py",
        "--latents_dir", str(latents_path),
    ]
    
    result = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("✗ Probe training failed!")
        return None
    
    # Find the latest probe directory
    probe_dir = PROJECT_ROOT / "checkpoints" / "probes"
    runs = sorted(probe_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("✗ No probe directory found!")
        return None
    
    probe_path = runs[0] / "pytorch"
    print(f"✓ Probes trained: {probe_path}")
    
    return str(probe_path)


def run_phase2(probe_path=None, config="configs/train_ppo_best.yaml"):
    """Run Phase 2: Train PPO policy."""
    print("\n" + "="*60)
    print("STEP 3: PHASE 2 - TRAINING PPO POLICY")
    print("="*60)
    
    if probe_path is None:
        probe_path = find_latest_probe()
        if probe_path is None:
            print("✗ No probe found! Please run Phase 1 first.")
            return None
        print(f"Using latest probe: {probe_path}")
    
    import shutil
    python_cmd = shutil.which("py")
    if python_cmd:
        python_exec = [python_cmd, "-3.11"]
    else:
        python_exec = [sys.executable]
    
    train_cmd = python_exec + [
        "scripts/train_ppo.py",
        "--config", config,
        "--probe_path", probe_path,
    ]
    
    result = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("✗ PPO training failed!")
        return None
    
    # Find the latest policy
    ppo_dir = PROJECT_ROOT / "outputs" / "ppo"
    runs = sorted(ppo_dir.glob("aether_ppo_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("✗ No policy directory found!")
        return None
    
    policy_path = runs[0] / "final_policy.pt"
    print(f"✓ Policy trained: {policy_path}")
    
    return str(policy_path)


def run_phase3(policy_path=None, probe_path=None, num_samples=50):
    """Run Phase 3: Evaluate policy."""
    print("\n" + "="*60)
    print("STEP 4: PHASE 3 - EVALUATING POLICY")
    print("="*60)
    
    if policy_path is None:
        ppo_dir = PROJECT_ROOT / "outputs" / "ppo"
        runs = sorted(ppo_dir.glob("aether_ppo_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("✗ No policy found! Please run Phase 2 first.")
            return None
        policy_path = runs[0] / "final_policy.pt"
        print(f"Using latest policy: {policy_path}")
    
    if probe_path is None:
        probe_path = find_latest_probe()
        if probe_path is None:
            print("✗ No probe found!")
            return None
        print(f"Using latest probe: {probe_path}")
    
    import shutil
    python_cmd = shutil.which("py")
    if python_cmd:
        python_exec = [python_cmd, "-3.11"]
    else:
        python_exec = [sys.executable]
    
    eval_cmd = python_exec + [
        "scripts/evaluate_ppo.py",
        "--policy_path", str(policy_path),
        "--probe_path", probe_path,
        "--num_samples", str(num_samples),
        "--device", "cuda",
    ]
    
    result = subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("✗ Evaluation failed!")
        return None
    
    print("✓ Evaluation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete Project Aether pipeline")
    parser.add_argument("--skip_setup", action="store_true", help="Skip setup verification")
    parser.add_argument("--skip_phase1", action="store_true", help="Skip Phase 1 (use existing probes)")
    parser.add_argument("--skip_phase2", action="store_true", help="Skip Phase 2 (use existing policy)")
    parser.add_argument("--skip_phase3", action="store_true", help="Skip Phase 3 (evaluation)")
    parser.add_argument("--full", action="store_true", help="Run everything from scratch")
    parser.add_argument("--config", default="configs/train_ppo_best.yaml", help="PPO config file")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for Phase 1")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of diffusion steps")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PROJECT AETHER - COMPLETE EXPERIMENT RUNNER")
    print("="*60)
    
    # Step 1: Setup verification
    if not args.skip_setup:
        if not run_setup_test():
            return 1
    
    # Step 2: Phase 1
    probe_path = None
    if not args.skip_phase1 or args.full:
        probe_path = run_phase1(args.num_samples, args.num_steps)
        if probe_path is None and not args.skip_phase1:
            return 1
    else:
        probe_path = find_latest_probe()
        if probe_path:
            print(f"\nUsing existing probe: {probe_path}")
        else:
            print("\n⚠ No existing probe found. Run Phase 1 first or use --full")
    
    # Step 3: Phase 2
    policy_path = None
    if not args.skip_phase2 or args.full:
        policy_path = run_phase2(probe_path, args.config)
        if policy_path is None and not args.skip_phase2:
            return 1
    else:
        ppo_dir = PROJECT_ROOT / "outputs" / "ppo"
        runs = sorted(ppo_dir.glob("aether_ppo_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            policy_path = runs[0] / "final_policy.pt"
            print(f"\nUsing existing policy: {policy_path}")
        else:
            print("\n⚠ No existing policy found. Run Phase 2 first or use --full")
    
    # Step 4: Phase 3
    if not args.skip_phase3:
        run_phase3(policy_path, probe_path, args.num_samples)
    
    print("\n" + "="*60)
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

