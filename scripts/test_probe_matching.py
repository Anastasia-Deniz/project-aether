#!/usr/bin/env python3
"""
Test script to validate probe matching logic.
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_all_policies import validate_probe_compatibility


def test_probe_compatibility():
    """Test the probe compatibility validation function."""

    # Test with a real policy
    project_root = Path(__file__).parent.parent
    policy_path = project_root / "outputs" / "ppo" / "aether_ppo_20251227_062856" / "final_policy.pt"

    if policy_path.exists():
        run_dir = policy_path.parent
        config_file = run_dir / "training_config.json"

        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

            probe_path = config.get('probe_path')
            if probe_path:
                print(f"Testing compatibility for policy: {policy_path.name}")
                print(f"Probe path from config: {probe_path}")

                is_compatible = validate_probe_compatibility(probe_path, config, project_root)
                print(f"Probe compatibility: {'✓ COMPATIBLE' if is_compatible else '✗ NOT COMPATIBLE'}")

                return is_compatible

    print("Could not find test policy or config")
    return False


if __name__ == "__main__":
    test_probe_compatibility()
