"""
Project Aether - Setup Test Script
Verifies that all dependencies are installed and working.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    imports = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("gymnasium", "Gymnasium"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
        ("PIL", "Pillow"),
        ("datasets", "HuggingFace Datasets"),
    ]
    
    failed = []
    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    try:
        import torch
    except ImportError:
        print("  ✗ PyTorch not installed, cannot test CUDA")
        return False
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ CUDA available")
        print(f"    Device: {device_name}")
        print(f"    Memory: {memory:.1f} GB")
        return True
    else:
        print("  ⚠ CUDA not available (will use CPU)")
        return False


def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")
    
    required = [
        "src/__init__.py",
        "src/envs/__init__.py",
        "src/envs/diffusion_env.py",
        "src/models/__init__.py",
        "src/models/linear_probe.py",
        "src/utils/__init__.py",
        "src/utils/data.py",
        "scripts/collect_latents.py",
        "scripts/train_probes.py",
        "configs/base.yaml",
        "requirements.txt",
    ]
    
    missing = []
    for path in required:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path}")
            missing.append(path)
    
    return len(missing) == 0, missing


def test_data_loading():
    """Test data loading utilities."""
    print("\nTesting data loading...")
    
    try:
        from src.utils.data import (
            DataConfig,
            I2PDataset,
            AlternativeSafePrompts,
        )
        print("  ✓ Data utilities imported")
        
        # Test safe prompts
        safe = AlternativeSafePrompts.get_prompts(5)
        print(f"  ✓ Safe prompts loaded ({len(safe)} samples)")
        
        # Test I2P (will try to download)
        print("  Testing I2P dataset (may download)...")
        config = DataConfig(num_unsafe_samples=5)
        i2p = I2PDataset(config)
        try:
            i2p.load()
            prompts = i2p.get_prompts(max_samples=5)
            print(f"  ✓ I2P dataset loaded ({len(prompts)} samples)")
        except Exception as e:
            print(f"  ⚠ I2P download failed (may need internet): {e}")
        
        return True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False


def test_environment():
    """Test the diffusion environment (quick check)."""
    print("\nTesting environment (quick check)...")
    
    try:
        from src.envs.diffusion_env import AetherConfig, DiffusionSteeringEnv
        print("  ✓ Environment classes imported")
        
        # Just test config creation, not full env (would download model)
        config = AetherConfig(
            num_inference_steps=20,  # Use realistic number of steps
            device="cpu",
        )
        print(f"  ✓ Config created (latent_dim={config.latent_dim})")
        
        return True
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        return False


def main():
    print("="*60)
    print("PROJECT AETHER - SETUP TEST")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['imports'], failed_imports = test_imports()
    results['cuda'] = test_cuda()
    results['structure'], missing_files = test_project_structure()
    results['data'] = test_data_loading()
    results['environment'] = test_environment()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")
        if not passed and test not in ['cuda']:  # CUDA is optional
            all_passed = False
    
    if failed_imports:
        print(f"\nMissing packages: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
    
    if not results['structure']:
        print(f"\nMissing files detected. Project may be incomplete.")
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run Phase 1.")
        print("\nNext step:")
        print("  python scripts/run_phase1.py --quick")
    else:
        print("\n⚠ Some tests failed. Please fix issues before proceeding.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

