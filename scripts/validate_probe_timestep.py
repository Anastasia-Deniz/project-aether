"""
Validate that the probe trained on final timestep (19) works correctly on final latents.
This validates that the probe timestep mismatch issue has been fixed.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.linear_probe import LinearProbe


def load_latents_and_labels(latents_dir: str, timestep: int = 19):
    """Load latents and labels for a specific timestep."""
    latents_dir = Path(latents_dir)
    latent_file = latents_dir / "latents" / f"timestep_{timestep:02d}.npz"

    if not latent_file.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_file}")

    # Load latents
    data = np.load(latent_file)
    X = data['latents']  # Shape: (num_samples, latent_dim)
    labels = data['labels']  # Shape: (num_samples,)

    return X, labels


def validate_probe_accuracy(probe_path: str, latents_dir: str, timestep: int = 19, device: str = 'cuda'):
    """Validate probe accuracy on the timestep it was trained on."""
    print(f"Validating probe trained on timestep {timestep}...")

    # Load probe
    probe = LinearProbe(input_dim=16384)  # SD latent dimension
    probe.load_state_dict(torch.load(probe_path, map_location=device))
    probe = probe.to(device)
    probe.eval()

    # Load test data (same timestep the probe was trained on)
    X_test, y_test = load_latents_and_labels(latents_dir, timestep)

    print(f"Test set: {len(X_test)} samples, {np.sum(y_test)} unsafe, {len(y_test) - np.sum(y_test)} safe")

    # Make predictions
    probe.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        logits = probe(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Calculate metrics
    preds = (probs > 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(".3f")
    print(".3f")
    print(".3f")

    return accuracy, auc


def main():
    # Paths
    latents_dir = "data/latents/run_20251225_182652"
    probe_dir = "checkpoints/probes/run_20251225_183438/pytorch"

    # Test probe trained on timestep 19 (final timestep)
    probe_path_t19 = Path(probe_dir) / "probe_t19.pt"

    if not probe_path_t19.exists():
        print(f"Error: Probe not found at {probe_path_t19}")
        return

    print("=== Validating Probe Timestep Match ===")
    print(f"Testing probe trained on timestep 19 against timestep 19 data")
    print()

    accuracy, auc = validate_probe_accuracy(str(probe_path_t19), latents_dir, timestep=19)

    print()
    print("=== Summary ===")
    if accuracy > 0.8 and auc > 0.9:
        print("✅ SUCCESS: Probe shows good accuracy on final timestep")
        print("   The probe timestep mismatch issue has been fixed!")
    elif accuracy > 0.7:
        print("⚠️  WARNING: Probe accuracy is acceptable but could be better")
        print("   Consider training with more data or different hyperparameters")
    else:
        print("❌ FAILURE: Probe accuracy is too low")
        print("   The probe timestep mismatch may still be an issue")


if __name__ == "__main__":
    main()
