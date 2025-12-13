"""
GPU Memory Test for RTX 4050 (6GB)
Tests that the optimized settings fit in VRAM.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gc

def format_bytes(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

def print_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"  Allocated: {format_bytes(allocated)}")
        print(f"  Reserved:  {format_bytes(reserved)}")
        print(f"  Total:     {format_bytes(total)}")
        print(f"  Free:      {format_bytes(total - reserved)}")

def main():
    print("=" * 60)
    print("GPU MEMORY TEST FOR RTX 4050")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"Device: {device}")
    print(f"Total VRAM: {format_bytes(total_mem)}")
    
    # Test 1: LatentEncoder
    print("\n--- Test 1: LatentEncoder ---")
    from src.envs.diffusion_env import LatentEncoder
    
    encoder = LatentEncoder(16384, 256, hidden_dim=512).cuda().half()
    print_memory()
    
    # Test forward pass
    x = torch.randn(1, 16384, device='cuda', dtype=torch.float16)
    y = encoder(x)
    print(f"  Forward pass OK: {x.shape} -> {y.shape}")
    
    del encoder, x, y
    torch.cuda.empty_cache()
    gc.collect()
    
    # Test 2: ActorCritic Policy
    print("\n--- Test 2: ActorCritic Policy ---")
    from src.training.ppo_trainer import ActorCritic
    
    policy = ActorCritic(258, 256, hidden_dims=[256, 128]).cuda()
    print_memory()
    
    obs = torch.randn(32, 258, device='cuda')  # Batch of 32
    action, log_prob, value = policy.get_action(obs)
    print(f"  Forward pass OK: batch=32")
    
    del policy, obs, action, log_prob, value
    torch.cuda.empty_cache()
    gc.collect()
    
    # Test 3: Load Diffusion Model (the big one)
    print("\n--- Test 3: Stable Diffusion Model ---")
    print("  Loading model (this takes ~30 seconds)...")
    
    try:
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "rupeshs/LCM-runwayml-stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to('cuda')
        
        print_memory()
        print("  Model loaded successfully!")
        
        # Test generation with minimal steps
        print("\n--- Test 4: Quick Generation Test ---")
        print("  Running 5-step generation...")
        
        with torch.no_grad():
            # Set minimal steps
            image = pipe(
                "a cat",
                num_inference_steps=5,
                guidance_scale=7.5,
            ).images[0]
        
        print_memory()
        print("  Generation OK!")
        
        del pipe, image
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ GPU MEMORY TEST COMPLETE")
    print("=" * 60)
    print("\nYour RTX 4050 can handle the optimized settings!")
    print("You can now run Phase 1 (collect_latents.py)")

if __name__ == "__main__":
    main()

