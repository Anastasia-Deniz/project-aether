"""
Project Aether - Phase 1 Runner
Complete pipeline to validate linear separability of unsafe concepts.

This script runs the full Phase 1 pipeline:
1. Load datasets (I2P unsafe, Safe prompts)
2. Collect latents at each timestep
3. Train linear probes
4. (Optional) Measure empirical layer sensitivity (FID & SSR) ⭐ NEW
5. Generate layer sensitivity analysis

Usage:
    python scripts/run_phase1.py --num_samples 50 --quick
    python scripts/run_phase1.py --num_samples 200 --measure_empirical --use_empirical
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1: Linear Probe Pipeline")
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples per class (safe/unsafe)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (fewer samples, no image saving)"
    )
    parser.add_argument(
        "--skip_collection",
        action="store_true",
        help="Skip latent collection (use existing data)"
    )
    parser.add_argument(
        "--latents_dir",
        type=str,
        default=None,
        help="Path to existing latents directory (for --skip_collection)"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images"
    )
    parser.add_argument(
        "--measure_empirical",
        action="store_true",
        help="Measure empirical layer sensitivity (FID & SSR) - takes additional time"
    )
    parser.add_argument(
        "--use_empirical",
        action="store_true",
        help="Use empirical measurements when training probes (requires --measure_empirical first)"
    )
    
    return parser.parse_args()


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {description}")
        sys.exit(1)
    
    print(f"\n✓ Completed: {description}")
    return result


def main():
    args = parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.num_samples = min(args.num_samples, 20)
        args.num_steps = min(args.num_steps, 10)
        print(f"Quick mode: {args.num_samples} samples, {args.num_steps} steps")
    
    print("="*60)
    print("PROJECT AETHER - PHASE 1: LINEAR PROBE PIPELINE")
    print("="*60)
    print(f"Samples per class: {args.num_samples}")
    print(f"Diffusion steps: {args.num_steps}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Collect Latents
    if not args.skip_collection:
        collect_cmd = [
            sys.executable, "scripts/collect_latents.py",
            "--num_samples", str(args.num_samples),
            "--num_steps", str(args.num_steps),
            "--device", args.device,
            "--seed", str(args.seed),
            "--output_dir", f"./data/latents",
            "--model_id", "CompVis/stable-diffusion-v1-4",
            "--focus_nudity",
            "--hard_only",
            "--min_nudity_pct", "50.0",
            "--min_inappropriate_pct", "60.0",
        ]
        
        if args.save_images:
            collect_cmd.append("--save_images")
        
        run_command(collect_cmd, "Latent Collection")
        
        # Find the latest run directory
        latents_base = PROJECT_ROOT / "data" / "latents"
        run_dirs = sorted(latents_base.glob("run_*"))
        if run_dirs:
            latents_dir = str(run_dirs[-1])
        else:
            print("❌ No latent directories found!")
            sys.exit(1)
    else:
        if args.latents_dir:
            latents_dir = args.latents_dir
        else:
            # Find the latest run directory
            latents_base = PROJECT_ROOT / "data" / "latents"
            run_dirs = sorted(latents_base.glob("run_*"))
            if run_dirs:
                latents_dir = str(run_dirs[-1])
            else:
                print("❌ No latent directories found! Run without --skip_collection first.")
                sys.exit(1)
    
    print(f"\nUsing latents from: {latents_dir}")
    
    # Step 2.5: (Optional) Measure Empirical Layer Sensitivity
    if args.measure_empirical:
        # Find probe directory if available
        probe_base = PROJECT_ROOT / "checkpoints" / "probes"
        probe_dirs = sorted(probe_base.glob("run_*"))
        probe_path = None
        if probe_dirs:
            probe_path = str(probe_dirs[-1] / "pytorch")
        
        measure_cmd = [
            sys.executable, "scripts/measure_layer_sensitivity.py",
            "--latents_dir", latents_dir,
            "--num_samples", "20",
            "--device", args.device,
        ]
        
        if probe_path and Path(probe_path).exists():
            measure_cmd.extend(["--probe_path", probe_path])
        
        run_command(measure_cmd, "Empirical Layer Sensitivity Measurement")
        print("\n✓ Empirical measurements saved. Re-running probe training with --use_empirical...")
    
    # Step 2: Train Probes
    probe_cmd = [
        sys.executable, "scripts/train_probes.py",
        "--latents_dir", latents_dir,
        "--output_dir", "./checkpoints/probes",
        "--seed", str(args.seed),
    ]
    
    if args.use_empirical or args.measure_empirical:
        probe_cmd.append("--use_empirical")
    
    run_command(probe_cmd, "Probe Training & Sensitivity Analysis")
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE")
    print("="*60)
    print(f"""
Results saved to:
  - Latents: {latents_dir}
  - Probes: ./checkpoints/probes/run_*/
  
Next steps:
  1. Review the probe_analysis.png plot
  2. Check if accuracy > 85% at any timestep
  3. If yes, proceed to Phase 2 (PPO training)
  4. Note the optimal intervention window from the analysis
  
Optional: For better accuracy, run empirical measurement:
  python scripts/measure_layer_sensitivity.py --latents_dir {latents_dir} --num_samples 20
  python scripts/train_probes.py --latents_dir {latents_dir} --use_empirical
  
To run Phase 2:
  python scripts/train_ppo.py --config configs/train_ppo_best.yaml
""")


if __name__ == "__main__":
    main()

