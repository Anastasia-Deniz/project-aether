"""
Test script to validate that the evaluation fixes work correctly.
Tests the simplified SSR/FPR logic and probe loading.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.linear_probe import LinearProbe
from src.evaluation.metrics import compute_ssr, compute_fpr


def test_probe_loading():
    """Test that probe_t19.pt can be loaded correctly."""
    print("Testing probe loading...")

    probe_path = "checkpoints/probes/run_20251225_183438/pytorch/probe_t19.pt"

    if not Path(probe_path).exists():
        print(f"❌ Probe file not found: {probe_path}")
        return False

    try:
        probe = LinearProbe(input_dim=16384)
        probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
        probe.eval()
        print("✅ Probe loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load probe: {e}")
        return False


def test_simplified_ssr_fpr_logic():
    """Test the simplified SSR/FPR calculation logic."""
    print("\nTesting simplified SSR/FPR logic...")

    # Mock data: 10 samples (5 safe, 5 unsafe)
    original_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Ground truth

    # Mock probe predictions on steered latents
    # Assume perfect probe: correctly predicts all unsafe as unsafe, all safe as safe
    steered_predictions = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Calculate SSR and FPR
    ssr, unsafe_to_safe, total_unsafe = compute_ssr(
        original_labels, steered_predictions, original_labels
    )

    fpr, safe_to_flagged, total_safe = compute_fpr(
        original_labels, steered_predictions, original_labels
    )

    print(f"SSR: {ssr:.3f} ({unsafe_to_safe}/{total_unsafe} unsafe converted)")
    print(f"FPR: {fpr:.3f} ({safe_to_flagged}/{total_safe} safe flagged)")

    # With perfect predictions, SSR should be 0 (no unsafe became safe)
    # FPR should be 0 (no safe were flagged as unsafe)
    expected_ssr = 0.0
    expected_fpr = 0.0

    if abs(ssr - expected_ssr) < 1e-6 and abs(fpr - expected_fpr) < 1e-6:
        print("✅ SSR/FPR calculations are correct")
        return True
    else:
        print(f"❌ SSR/FPR mismatch. Expected SSR={expected_ssr}, FPR={expected_fpr}")
        return False


def test_probe_predictions():
    """Test that probe makes reasonable predictions."""
    print("\nTesting probe predictions...")

    probe_path = "checkpoints/probes/run_20251225_183438/pytorch/probe_t19.pt"
    latents_path = "data/latents/run_20251225_182652/latents/timestep_19.npz"

    if not Path(probe_path).exists():
        print(f"❌ Probe file not found: {probe_path}")
        return False

    if not Path(latents_path).exists():
        print(f"❌ Latents file not found: {latents_path}")
        return False

    try:
        # Load probe
        probe = LinearProbe(input_dim=16384)
        probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
        probe.eval()

        # Load some test latents
        data = np.load(latents_path)
        X_test = data['latents'][:10]  # First 10 samples
        y_test = data['labels'][:10]

        # Make predictions
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_test).float()
            logits = probe(X_tensor)
            probs = torch.sigmoid(logits).numpy().flatten()
            preds = (probs > 0.5).astype(int)

        # Calculate accuracy
        accuracy = np.mean(preds == y_test)
        print(".3f")

        if accuracy > 0.7:
            print("✅ Probe shows good accuracy on final timestep")
            return True
        else:
            print("⚠️  Probe accuracy is low - may indicate remaining issues")
            return False

    except Exception as e:
        print(f"❌ Error testing probe predictions: {e}")
        return False


def main():
    print("=== Testing Evaluation Fixes ===\n")

    all_passed = True

    # Test 1: Probe loading
    if not test_probe_loading():
        all_passed = False

    # Test 2: Simplified SSR/FPR logic
    if not test_simplified_ssr_fpr_logic():
        all_passed = False

    # Test 3: Probe predictions
    if not test_probe_predictions():
        all_passed = False

    print(f"\n=== Summary ===")
    if all_passed:
        print("✅ All tests passed! Evaluation fixes are working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    return all_passed


if __name__ == "__main__":
    main()
