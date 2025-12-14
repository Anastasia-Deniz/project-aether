"""
Switch all configs from LCM SD 1.5 to CompVis SD 1.4.

SD 1.4 is less censored and better for generating unsafe content for training.
Note: SD 1.4 requires more steps (20-50) than LCM (4-8).
"""

import sys
from pathlib import Path
import yaml

def update_config_file(config_path: Path, old_model: str, new_model: str, old_steps: int, new_steps: int):
    """Update a single config file."""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace model ID
    if old_model in content:
        content = content.replace(old_model, new_model)
        print(f"  ✓ Updated model in {config_path.name}")
    
    # Update inference steps if present
    if f"num_inference_steps: {old_steps}" in content:
        content = content.replace(f"num_inference_steps: {old_steps}", f"num_inference_steps: {new_steps}")
        print(f"  ✓ Updated steps to {new_steps} in {config_path.name}")
    elif "num_inference_steps:" in content and old_steps in content:
        # Handle different formats
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "num_inference_steps:" in line and str(old_steps) in line:
                lines[i] = line.replace(str(old_steps), str(new_steps))
                print(f"  ✓ Updated steps to {new_steps} in {config_path.name}")
        content = '\n'.join(lines)
    
    with open(config_path, 'w') as f:
        f.write(content)
    
    return old_model in content or f"num_inference_steps: {old_steps}" in content

def main():
    old_model = "rupeshs/LCM-runwayml-stable-diffusion-v1-5"
    new_model = "CompVis/stable-diffusion-v1-4"
    old_steps = 8  # LCM uses 4-8 steps
    new_steps = 20  # SD 1.4 uses 20-50 steps
    
    print("="*60)
    print("Switching to Stable Diffusion 1.4 (CompVis)")
    print("="*60)
    print(f"Old model: {old_model}")
    print(f"New model: {new_model}")
    print(f"Old steps: {old_steps}")
    print(f"New steps: {new_steps}")
    print()
    
    # Find all config files
    config_dir = Path("configs")
    config_files = list(config_dir.rglob("*.yaml"))
    
    updated = 0
    for config_file in config_files:
        if update_config_file(config_file, old_model, new_model, old_steps, new_steps):
            updated += 1
    
    print(f"\n✓ Updated {updated} config files")
    print("\nNote: SD 1.4 requires more steps (20+) than LCM (4-8)")
    print("      This will be slower but generates less censored content.")

if __name__ == "__main__":
    main()

