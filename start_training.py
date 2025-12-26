"""
Quick script to start PPO training with existing probes.
Handles path encoding issues on Windows.
"""

import sys
import subprocess
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Find latest probe
probe_dir = PROJECT_ROOT / "checkpoints" / "probes"
runs = sorted(probe_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)

if not runs:
    print("❌ No probes found! Please run Phase 1 first.")
    sys.exit(1)

latest_probe = runs[0] / "pytorch"
print(f"✓ Using probe: {latest_probe}")
print(f"✓ Probe run: {runs[0].name}")

# Use py -3.11 to match setup_env.bat
import shutil
python_cmd = shutil.which("py")
if python_cmd:
    python_exec = [python_cmd, "-3.11"]
else:
    python_exec = [sys.executable]

# Build command
cmd = python_exec + [
    str(PROJECT_ROOT / "scripts" / "train_ppo.py"),
    "--config", str(PROJECT_ROOT / "configs" / "train_ppo_best.yaml"),
    "--probe_path", str(latest_probe),
]

print(f"\n{'='*60}")
print("STARTING PPO TRAINING")
print(f"{'='*60}")
print(f"Command: {' '.join(cmd)}")
print(f"{'='*60}\n")

# Run training
result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

if result.returncode == 0:
    print("\n✓ Training completed successfully!")
else:
    print("\n✗ Training failed!")
    sys.exit(result.returncode)

