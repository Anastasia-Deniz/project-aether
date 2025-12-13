#!/usr/bin/env python3
"""
Project Aether - Teammate Setup Script
Run this script to set up your environment and verify everything works.

Usage:
    python setup_teammate.py
    
    # Or with options:
    python setup_teammate.py --skip-model  # Skip model download
    python setup_teammate.py --cpu         # Force CPU mode
"""

import sys
import subprocess
import argparse
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)

def print_ok(text):
    print(f"  ✓ {text}")

def print_fail(text):
    print(f"  ✗ {text}")

def print_warn(text):
    print(f"  ⚠ {text}")

def run_command(cmd, capture=True):
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture, 
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_python():
    print_header("Checking Python")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print_ok("Python version OK")
        return True
    else:
        print_fail("Python 3.10+ required")
        return False

def check_pytorch():
    print_header("Checking PyTorch")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print_ok("GPU mode enabled")
            return "cuda"
        else:
            print(f"  CUDA available: No")
            print_warn("Running in CPU mode (slow)")
            return "cpu"
    except ImportError:
        print_fail("PyTorch not installed")
        print("  Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return None

def check_dependencies():
    print_header("Checking Dependencies")
    
    required = [
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("gymnasium", "Gymnasium"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
        ("PIL", "Pillow"),
        ("datasets", "HuggingFace Datasets"),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print_ok(name)
        except ImportError:
            print_fail(f"{name} not installed")
            missing.append(name)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    return True

def check_project_structure():
    print_header("Checking Project Structure")
    
    required_files = [
        "src/__init__.py",
        "src/envs/__init__.py",
        "src/envs/diffusion_env.py",
        "src/models/__init__.py",
        "src/models/linear_probe.py",
        "src/training/__init__.py",
        "src/training/ppo_trainer.py",
        "src/evaluation/__init__.py",
        "src/evaluation/metrics.py",
        "src/utils/__init__.py",
        "src/utils/data.py",
        "scripts/collect_latents.py",
        "scripts/train_probes.py",
        "scripts/run_sensitivity.py",
        "scripts/train_ppo.py",
        "configs/base.yaml",
        "configs/train_ppo.yaml",
        "requirements.txt",
    ]
    
    missing = []
    for f in required_files:
        if Path(f).exists():
            print_ok(f)
        else:
            print_fail(f"{f} not found")
            missing.append(f)
    
    if missing:
        print(f"\n  Missing files: {len(missing)}")
        return False
    return True

def test_components():
    print_header("Testing Components")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Test 1: LatentEncoder
    try:
        from src.envs.diffusion_env import LatentEncoder
        import torch
        enc = LatentEncoder(16384, 256)
        x = torch.randn(1, 16384)
        y = enc(x)
        assert y.shape == (1, 256)
        print_ok("LatentEncoder")
    except Exception as e:
        print_fail(f"LatentEncoder: {e}")
        return False
    
    # Test 2: PPO Components
    try:
        from src.training.ppo_trainer import ActorCritic
        policy = ActorCritic(258, 256)
        obs = torch.randn(4, 258)
        action, _, _ = policy.get_action(obs)
        assert action.shape == (4, 256)
        print_ok("PPO ActorCritic")
    except Exception as e:
        print_fail(f"PPO ActorCritic: {e}")
        return False
    
    # Test 3: Evaluation Metrics
    try:
        from src.evaluation.metrics import compute_ssr
        import numpy as np
        ssr, _, _ = compute_ssr(
            np.array([1, 1, 0]),
            np.array([0, 1, 0]),
            np.array([1, 1, 0])
        )
        assert 0 <= ssr <= 1
        print_ok("Evaluation Metrics")
    except Exception as e:
        print_fail(f"Evaluation Metrics: {e}")
        return False
    
    # Test 4: Safe Prompts
    try:
        from src.utils.data import AlternativeSafePrompts
        prompts = AlternativeSafePrompts.get_prompts(10)
        assert len(prompts) == 10
        print_ok(f"Safe Prompts ({AlternativeSafePrompts.get_base_count()} base)")
    except Exception as e:
        print_fail(f"Safe Prompts: {e}")
        return False
    
    return True

def download_model():
    print_header("Downloading Model")
    print("  This will download ~4GB from HuggingFace")
    print("  (This is a one-time download)")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
        print("  Loading model (may take 15-30 minutes)...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "rupeshs/LCM-runwayml-stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        print_ok("Model downloaded and cached")
        return True
    except Exception as e:
        print_fail(f"Model download failed: {e}")
        print("  You can download later by running:")
        print("  python -c \"from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('rupeshs/LCM-runwayml-stable-diffusion-v1-5')\"")
        return False

def main():
    parser = argparse.ArgumentParser(description="Project Aether Setup")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  PROJECT AETHER - TEAMMATE SETUP")
    print("="*60)
    
    results = {}
    
    # Check Python
    results['python'] = check_python()
    
    # Check PyTorch
    device = check_pytorch()
    results['pytorch'] = device is not None
    
    if args.cpu:
        device = "cpu"
        print("  (Forced CPU mode)")
    
    # Check dependencies
    results['dependencies'] = check_dependencies()
    
    # Check project structure
    results['structure'] = check_project_structure()
    
    # Test components
    if results['pytorch'] and results['dependencies'] and results['structure']:
        results['components'] = test_components()
    else:
        results['components'] = False
        print_header("Testing Components")
        print_warn("Skipped - fix previous issues first")
    
    # Download model
    if not args.skip_model and all(results.values()):
        results['model'] = download_model()
    else:
        results['model'] = True  # Not required
        if args.skip_model:
            print_header("Model Download")
            print_warn("Skipped (--skip-model flag)")
    
    # Summary
    print_header("SETUP SUMMARY")
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        if passed:
            print_ok(name)
        else:
            print_fail(name)
    
    if all_passed:
        print("\n" + "="*60)
        print("  ✓ SETUP COMPLETE - Ready to run!")
        print("="*60)
        print("\n  Next steps:")
        print("  1. Run: python scripts/quick_test.py")
        print("  2. Start Phase 1: python scripts/collect_latents.py --num_samples 50")
        print(f"\n  Device: {device.upper()}")
    else:
        print("\n" + "="*60)
        print("  ✗ SETUP INCOMPLETE - Fix issues above")
        print("="*60)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

