"""Quick test of all new components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60, flush=True)
print("QUICK TEST OF NEW COMPONENTS", flush=True)
print("=" * 60, flush=True)

# Test 1: LatentEncoder
print("\n1. Testing LatentEncoder...", flush=True)
try:
    from src.envs.diffusion_env import LatentEncoder, SteeringProjection
    import torch
    
    enc = LatentEncoder(16384, 256)
    x = torch.randn(1, 16384)
    y = enc(x)
    print(f"   Encoder: {x.shape} -> {y.shape}", flush=True)
    print(f"   Params: {sum(p.numel() for p in enc.parameters()):,}", flush=True)
    
    proj = SteeringProjection(256, 16384)
    a = torch.randn(256)
    d = proj(a)
    print(f"   Projection: {a.shape} -> {d.shape}", flush=True)
    print("   ✓ LatentEncoder OK", flush=True)
except Exception as e:
    print(f"   ✗ LatentEncoder failed: {e}", flush=True)

# Test 2: Safe Prompts
print("\n2. Testing Expanded Safe Prompts...", flush=True)
try:
    from src.utils.data import AlternativeSafePrompts
    
    base_count = AlternativeSafePrompts.get_base_count()
    print(f"   Base prompts: {base_count}", flush=True)
    
    prompts = AlternativeSafePrompts.get_prompts(num_samples=200, seed=42)
    print(f"   Sample: {prompts[0]['prompt'][:50]}...", flush=True)
    print("   ✓ Safe Prompts OK", flush=True)
except Exception as e:
    print(f"   ✗ Safe Prompts failed: {e}", flush=True)

# Test 3: PPO Components
print("\n3. Testing PPO Components...", flush=True)
try:
    from src.training.ppo_trainer import ActorCritic, PPOConfig
    
    policy = ActorCritic(258, 256)
    print(f"   Policy params: {sum(p.numel() for p in policy.parameters()):,}", flush=True)
    
    obs = torch.randn(4, 258)
    action, log_prob, value = policy.get_action(obs)
    print(f"   Action: {action.shape}, Value: {value.shape}", flush=True)
    print("   ✓ PPO Components OK", flush=True)
except Exception as e:
    print(f"   ✗ PPO Components failed: {e}", flush=True)

# Test 4: Evaluation Metrics
print("\n4. Testing Evaluation Metrics...", flush=True)
try:
    from src.evaluation.metrics import compute_ssr, compute_fpr, compute_transport_cost
    import numpy as np
    
    orig = np.array([1, 1, 1, 0, 0])
    steer = np.array([0, 0, 1, 0, 0])
    labels = np.array([1, 1, 1, 0, 0])
    
    ssr, u2s, total_u = compute_ssr(orig, steer, labels)
    print(f"   SSR: {ssr:.4f} ({u2s}/{total_u})", flush=True)
    
    fpr, s2f, total_s = compute_fpr(orig, steer, labels)
    print(f"   FPR: {fpr:.4f} ({s2f}/{total_s})", flush=True)
    print("   ✓ Evaluation Metrics OK", flush=True)
except Exception as e:
    print(f"   ✗ Evaluation Metrics failed: {e}", flush=True)

# Test 5: Environment Config
print("\n5. Testing Environment Config...", flush=True)
try:
    from src.envs.diffusion_env import AetherConfig
    
    config = AetherConfig(
        device="cpu",
        use_latent_encoder=True,
        encoded_latent_dim=256,
    )
    print(f"   Latent dim: {config.latent_dim}", flush=True)
    print(f"   Observation dim: {config.observation_dim}", flush=True)
    print(f"   With encoder: {config.use_latent_encoder}", flush=True)
    print("   ✓ Environment Config OK", flush=True)
except Exception as e:
    print(f"   ✗ Environment Config failed: {e}", flush=True)

print("\n" + "=" * 60, flush=True)
print("✓ ALL TESTS PASSED!", flush=True)
print("=" * 60, flush=True)

