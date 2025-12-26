"""
Project Aether - Diffusion Environment for RL
Wraps a frozen diffusion model as a Gymnasium environment.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from diffusers import StableDiffusionPipeline
from dataclasses import dataclass, field


class LatentEncoder(nn.Module):
    """
    Encoder to reduce high-dimensional latent observations to a manageable size.
    
    This addresses the problem of PPO struggling with 16K+ dimensional observations.
    Uses a simple MLP with optional layer normalization for stability.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 256,
        hidden_dim: int = 1024,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        ]
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode high-dimensional latent to low-dimensional representation."""
        return self.encoder(x)


class SteeringProjection(nn.Module):
    """
    Projects low-dimensional actions to full latent space.
    
    Can be initialized randomly (fixed), with PCA, or learned alongside policy.
    """
    
    def __init__(
        self, 
        action_dim: int, 
        latent_dim: int,
        init_method: str = "orthogonal",  # "orthogonal", "random", "learned"
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        self.projection = nn.Linear(action_dim, latent_dim, bias=False)
        
        # Initialize projection matrix
        if init_method == "orthogonal":
            nn.init.orthogonal_(self.projection.weight)
        elif init_method == "random":
            nn.init.normal_(self.projection.weight, std=0.01)
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Project action to latent space."""
        return self.projection(action)


@dataclass
class AetherConfig:
    """
    Configuration for the Aether environment.
    Default values optimized for RTX 4050 (6GB VRAM).
    """
    # Using Stable Diffusion 1.4 (CompVis) - less censored than SD 1.5
    # SD 1.4 requires 20-50 steps (slower than LCM but better for unsafe content)
    model_id: str = "CompVis/stable-diffusion-v1-4"
    num_inference_steps: int = 20  # SD 1.4 uses 20-50 steps
    guidance_scale: float = 7.5
    latent_channels: int = 4
    latent_size: int = 64
    steering_dim: int = 256  # Low-rank steering subspace
    lambda_transport: float = 0.5  # λ in the reward
    device: str = "cuda"
    dtype: torch.dtype = torch.float16  # CRITICAL: Half precision for memory
    
    # Intervention window (adjusted for 20 steps with SD 1.4)
    # Scaled from [2, 6] for 8 steps to [5, 15] for 20 steps (~25% to 75%)
    intervention_start: int = 5   # ~25% of generation
    intervention_end: int = 15    # ~75% of generation
    
    # Action constraints
    max_action_norm: float = 0.1  # Clip steering magnitude
    
    # Latent encoder settings (reduces observation from 16K to 256)
    use_latent_encoder: bool = True
    encoded_latent_dim: int = 256
    encoder_hidden_dim: int = 512  # Reduced from 1024 for memory
    
    # Steering projection initialization
    steering_init: str = "orthogonal"  # "orthogonal", "random", "learned"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        errors = []
        
        # Validate intervention window
        if self.intervention_start < 0:
            errors.append(f"intervention_start ({self.intervention_start}) must be >= 0")
        if self.intervention_end > self.num_inference_steps:
            errors.append(f"intervention_end ({self.intervention_end}) must be <= num_inference_steps ({self.num_inference_steps})")
        if self.intervention_start >= self.intervention_end:
            errors.append(f"intervention_start ({self.intervention_start}) must be < intervention_end ({self.intervention_end})")
        
        # Validate lambda
        if self.lambda_transport < 0:
            errors.append(f"lambda_transport ({self.lambda_transport}) must be >= 0")
        
        # Validate steering dimension
        if self.steering_dim <= 0:
            errors.append(f"steering_dim ({self.steering_dim}) must be > 0")
        
        # Validate action norm
        if self.max_action_norm <= 0:
            errors.append(f"max_action_norm ({self.max_action_norm}) must be > 0")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
    
    @property
    def latent_dim(self) -> int:
        """Total flattened latent dimension."""
        return self.latent_channels * self.latent_size * self.latent_size
    
    @property
    def observation_dim(self) -> int:
        """Dimension of the observation space."""
        if self.use_latent_encoder:
            return self.encoded_latent_dim + 1 + 1  # encoded_z + t + score
        else:
            return self.latent_dim + 1 + 1  # flat_z + t + score


class DiffusionSteeringEnv(gym.Env):
    """
    Gymnasium environment that wraps the diffusion ODE.
    
    The agent observes (z_t, t, safety_score) and outputs steering vectors.
    Episodes run backwards from t=1 (noise) to t=0 (image).
    
    Key equations from the Aether framework:
    - ODE Dynamics: dz/dt = v_θ(z_t, t)
    - Steering: z̄_t = z_t + π_φ(z_t, t)
    - Reward: J(φ) = E[R_safe - λ Σ_t ||a_t||²]
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self,
        config: AetherConfig,
        safety_classifier: Optional[torch.nn.Module] = None,
        linear_probe: Optional[torch.nn.Module] = None,
        prompts: Optional[List[str]] = None,
        load_model: bool = True,  # Set False for quick testing without GPU
    ):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Prompt pool for training
        self.prompts = prompts or ["a photo of a person"]
        self.current_prompt_idx = 0
        
        # Load frozen base model
        self.pipe = None
        if load_model:
            print(f"Loading base model: {config.model_id}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                config.model_id,
                torch_dtype=config.dtype,
                safety_checker=None,  # We'll use our own
            ).to(self.device)
            
            # Freeze the base model
            self.pipe.unet.eval()
            for param in self.pipe.unet.parameters():
                param.requires_grad = False
            
            # Also freeze VAE and text encoder
            self.pipe.vae.eval()
            for param in self.pipe.vae.parameters():
                param.requires_grad = False
            
        # Safety components
        self.safety_classifier = safety_classifier
        self.linear_probe = linear_probe
        
        # Latent encoder for reduced observation space (NEW)
        self.latent_encoder = None
        if config.use_latent_encoder:
            self.latent_encoder = LatentEncoder(
                input_dim=config.latent_dim,
                output_dim=config.encoded_latent_dim,
                hidden_dim=config.encoder_hidden_dim,
            ).to(self.device)
            # Freeze encoder initially (can be unfrozen for joint training)
            for param in self.latent_encoder.parameters():
                param.requires_grad = False
        
        # Define observation space
        obs_dim = config.observation_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: low-rank steering vector
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.steering_dim,), dtype=np.float32
        )
        
        # Steering projection (improved version)
        self.steering_proj = SteeringProjection(
            action_dim=config.steering_dim,
            latent_dim=config.latent_dim,
            init_method=config.steering_init,
        ).to(self.device)
        
        # Convert to appropriate dtype
        if config.dtype == torch.float16:
            self.steering_proj = self.steering_proj.half()
        
        # Episode state
        self.current_latent = None
        self.current_step = 0
        self.prompt = None
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.trajectory = []  # Store for analysis
        self.actions_taken = []  # Stores (timestep, action_norm_sq) tuples for weighted cost
        self.sensitivity_weights = self._load_sensitivity_weights()
        self.previous_probe_score = None  # For intermediate rewards
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to a new episode with a prompt."""
        super().reset(seed=seed)
        
        # MEMORY OPTIMIZATION: Clear previous episode data and CUDA cache
        if self.current_latent is not None:
            del self.current_latent
        if self.trajectory:
            del self.trajectory
        if self.actions_taken:
            del self.actions_taken
        if self.prompt_embeds is not None:
            del self.prompt_embeds
        if self.negative_prompt_embeds is not None:
            del self.negative_prompt_embeds
        
        # Clear CUDA cache to free memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Get prompt from options or cycle through prompts
        if options and "prompt" in options:
            self.prompt = options["prompt"]
        else:
            self.prompt = self.prompts[self.current_prompt_idx % len(self.prompts)]
            self.current_prompt_idx += 1
        
        # Encode prompt - handle different diffusers versions
        encode_result = self.pipe.encode_prompt(
            self.prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.config.guidance_scale > 1.0,
        )
        
        # Unpack based on return format
        if isinstance(encode_result, tuple):
            if len(encode_result) >= 2:
                self.prompt_embeds = encode_result[0]
                self.negative_prompt_embeds = encode_result[1]
            else:
                self.prompt_embeds = encode_result[0]
                self.negative_prompt_embeds = None
        else:
            self.prompt_embeds = encode_result
            self.negative_prompt_embeds = None
        
        # Set scheduler timesteps - CRITICAL FIX
        self.pipe.scheduler.set_timesteps(
            self.config.num_inference_steps, 
            device=self.device
        )
        
        # Initialize latent from noise
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            
        self.current_latent = torch.randn(
            1, self.config.latent_channels, 
            self.config.latent_size, self.config.latent_size,
            device=self.device,
            dtype=self.config.dtype,
            generator=generator,
        )
        
        # Scale initial noise by scheduler
        self.current_latent = self.current_latent * self.pipe.scheduler.init_noise_sigma
        
        # Reset counters
        self.current_step = 0
        self.trajectory = []  # MEMORY OPTIMIZATION: Don't store trajectory during training
        self.actions_taken = []  # Stores (timestep, action_norm_sq) tuples
        self.previous_probe_score = None  # Reset for intermediate rewards
        
        # Get initial observation
        obs = self._get_observation()
        info = {
            "prompt": self.prompt, 
            "timestep": 1.0,
            "step": 0,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the steered ODE.
        
        Args:
            action: Low-rank steering vector (will be projected to full latent dim)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Check if we should intervene at this timestep
        should_intervene = (
            self.config.intervention_start <= self.current_step <= self.config.intervention_end
        )
        
        # Convert action to steering vector using projection module
        action_tensor = torch.from_numpy(action).to(self.device).to(self.config.dtype)
        
        if should_intervene:
            with torch.no_grad():
                delta_z = self.steering_proj(action_tensor)
            delta_z = delta_z.view(
                1, self.config.latent_channels, 
                self.config.latent_size, self.config.latent_size
            )
            
            # Clip action magnitude to prevent image destruction
            action_norm = delta_z.norm()
            if action_norm > self.config.max_action_norm:
                delta_z = delta_z * (self.config.max_action_norm / action_norm)
            
            # Apply steering (Equation 5 in your doc): z̄_t = z_t + Δz
            steered_latent = self.current_latent + delta_z
        else:
            steered_latent = self.current_latent
            delta_z = torch.zeros_like(self.current_latent)
        
        # Get timestep for scheduler
        t = self.pipe.scheduler.timesteps[self.current_step]
        
        # Prepare latent for UNet (classifier-free guidance)
        if self.config.guidance_scale > 1.0 and self.negative_prompt_embeds is not None:
            latent_model_input = torch.cat([steered_latent] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Concatenate prompt embeddings
            prompt_embeds_combined = torch.cat([
                self.negative_prompt_embeds, 
                self.prompt_embeds
            ])
        else:
            latent_model_input = self.pipe.scheduler.scale_model_input(steered_latent, t)
            prompt_embeds_combined = self.prompt_embeds
        
        # Run one ODE step (Equation 6)
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_combined,
                return_dict=False,
            )[0]
        
        # Apply classifier-free guidance
        if self.config.guidance_scale > 1.0 and self.negative_prompt_embeds is not None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        
        # Scheduler step
        self.current_latent = self.pipe.scheduler.step(
            noise_pred, t, steered_latent, return_dict=False
        )[0]
        
        # Track trajectory - MEMORY OPTIMIZATION: Only store if needed for analysis
        # Skip trajectory storage during training to save VRAM
        # self.trajectory.append(self.current_latent.clone())
        
        # Only store action norms for transport cost (not full tensors)
        if should_intervene:
            # Store (timestep, action_norm_sq) for weighted cost
            action_norm_sq = delta_z.pow(2).sum().item()
            # Map current_step to timestep index (0-7 for 8-step, 0-19 for 20-step)
            timestep_idx = self.current_step
            self.actions_taken.append((timestep_idx, action_norm_sq))
        
        # Advance step counter
        self.current_step += 1
        terminated = self.current_step >= self.config.num_inference_steps
        
        # Compute reward with intermediate shaping
        if terminated:
            reward = self._compute_final_reward()
        else:
            # Intermediate reward: small bonus for moving toward safety
            reward = self._compute_intermediate_reward()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute transport cost so far (actions_taken now stores scalar norms)
        transport_cost = (
            sum(norm_sq for _, norm_sq in self.actions_taken)
            if self.actions_taken
            else 0.0
        )
        
        info = {
            "timestep": 1.0 - (self.current_step / self.config.num_inference_steps),
            "step": self.current_step,
            "transport_cost": transport_cost,
            "intervened": should_intervene,
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        # Get latent representation
        z_flat = self.current_latent.flatten().float()
        
        # Normalize latent to prevent numerical issues (zero mean, unit variance)
        z_mean = z_flat.mean()
        z_std = z_flat.std() + 1e-8  # Prevent division by zero
        z_normalized = (z_flat - z_mean) / z_std
        
        # Encode latent if encoder is available (reduces from 16K to 256 dims)
        if self.latent_encoder is not None:
            with torch.no_grad():
                z_encoded = self.latent_encoder(z_normalized.unsqueeze(0).to(self.device))
                z_obs = z_encoded.squeeze(0).cpu().numpy()
        else:
            z_obs = z_normalized.cpu().numpy()
        
        # Clamp to prevent extreme values and replace NaN with 0
        z_obs = np.clip(z_obs, -10.0, 10.0)
        z_obs = np.nan_to_num(z_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Normalized timestep
        t = np.array([1.0 - (self.current_step / self.config.num_inference_steps)])
        
        # Safety score from linear probe (use original unnormalized latent for probe)
        if self.linear_probe is not None:
            with torch.no_grad():
                score = self.linear_probe(z_flat.unsqueeze(0).to(self.device))
                score = torch.sigmoid(score).cpu().numpy().flatten()
                score = np.nan_to_num(score, nan=0.5)  # Default to neutral if NaN
        else:
            score = np.array([0.5])  # Neutral if no probe
        
        return np.concatenate([z_obs, t, score]).astype(np.float32)
    
    def _compute_final_reward(self) -> float:
        """
        Compute the reward: R_safe - λ * Σ||a_t||²
        
        This is Equation 7 from your document:
        J(φ) = E[R_safe - λ Σ_t ||a_t||²]
        """
        # Decode final latent to image
        with torch.no_grad():
            image = self.pipe.vae.decode(
                self.current_latent / self.pipe.vae.config.scaling_factor,
                return_dict=False
            )[0]
            image = (image / 2 + 0.5).clamp(0, 1)
        
        # Safety reward
        if self.safety_classifier is not None:
            with torch.no_grad():
                safety_score = self.safety_classifier(image)
                r_safe = safety_score.item()  # +1 safe, -1 unsafe
        else:
            # Fallback: use linear probe on final latent
            if self.linear_probe is not None:
                with torch.no_grad():
                    logit = self.linear_probe(self.current_latent.flatten().float().unsqueeze(0))
                    prob_safe = torch.sigmoid(-logit).item()  # Negative because probe predicts unsafe
                    r_safe = 2 * prob_safe - 1  # Map [0,1] to [-1,1]
            else:
                r_safe = 0.0  # No classifier, neutral reward
        
        # Transport cost (Wasserstein-inspired: Σ||Δz_t||²)
        # Use sensitivity-weighted cost if available
        if self.sensitivity_weights and self.actions_taken:
            # Weight actions by sensitivity scores
            weighted_cost = 0.0
            for timestep_idx, action_norm_sq in self.actions_taken:
                # Map timestep to sensitivity weight (scale from 8-step to current num_steps)
                # Sensitivity analysis was for 8 steps, we have 20 steps
                sensitivity_t = int(timestep_idx * 8 / self.config.num_inference_steps)
                weight = self.sensitivity_weights.get(str(sensitivity_t), 1.0)
                # Lower weight = more sensitive = less penalty (inverse relationship)
                # So we divide by weight (higher sensitivity = lower penalty)
                weighted_cost += action_norm_sq / (weight + 0.1)  # +0.1 to avoid division by zero
            transport_cost = weighted_cost
        else:
            # Fallback: simple sum
            transport_cost = sum(norm_sq for _, norm_sq in self.actions_taken) if self.actions_taken else 0.0
        
        # Final reward: R_safe - λ * W2_cost
        # Add efficiency bonus if achieved safety with low cost
        efficiency_bonus = 0.0
        if r_safe > 0 and transport_cost < 20.0:  # Achieved safety efficiently
            efficiency_bonus = 0.3  # Increased bonus for faster learning
        
        # Add bonus for strong safety improvement (unsafe -> very safe)
        safety_bonus = 0.0
        if r_safe > 0.5:  # Very safe (prob_safe > 0.75)
            safety_bonus = 0.2
        
        reward = r_safe - self.config.lambda_transport * transport_cost + efficiency_bonus + safety_bonus
        
        return reward
    
    def _load_sensitivity_weights(self) -> Optional[dict]:
        """Load sensitivity weights from Phase 1 analysis."""
        import json
        from pathlib import Path
        
        # Find the most recent sensitivity scores file
        candidates = sorted(
            Path("checkpoints/probes").glob("run_*/sensitivity_scores.json"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        for path in candidates:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    weights = {
                        t: info.get('score', 1.0)
                        for t, info in data.items()
                        if t != "optimal_window" and isinstance(info, dict)
                    }
                    if weights:
                        print(f"Loaded sensitivity weights from {path}")
                        return weights
            except Exception as e:
                print(f"Warning: Could not load sensitivity weights from {path}: {e}")
        
        return None
    
    def _compute_intermediate_reward(self) -> float:
        """
        Compute intermediate reward for moving toward safety.
        
        Improved reward shaping:
        - Probe improvement reward (moving toward safety)
        - Transport cost penalty (encourages efficient steering)
        - Safety maintenance bonus
        """
        if self.linear_probe is None:
            # If no probe, only penalize large actions
            if self.actions_taken:
                last_action_norm_sq = self.actions_taken[-1][1]
                return -0.01 * self.config.lambda_transport * last_action_norm_sq
            return 0.0
        
        # Get current probe score
        with torch.no_grad():
            z_flat = self.current_latent.flatten().float()
            current_score = self.linear_probe(z_flat.unsqueeze(0).to(self.device))
            current_prob_safe = torch.sigmoid(-current_score).item()
        
        # Reward for moving toward safety (increasing prob_safe)
        probe_reward = 0.0
        if self.previous_probe_score is not None:
            improvement = current_prob_safe - self.previous_probe_score
            # Reward for improvement (0.1 max per step)
            probe_reward = 0.1 * max(0, improvement)
            
            # Bonus if we're already safe and maintaining it
            if current_prob_safe > 0.7:
                probe_reward += 0.05
        else:
            probe_reward = 0.0
        
        # Transport cost penalty for current step (if action was taken)
        transport_penalty = 0.0
        if self.actions_taken:
            # Get the most recent action norm
            last_action_norm_sq = self.actions_taken[-1][1]
            # Small penalty per step to encourage efficiency
            # Scale by lambda_transport to match final reward structure
            transport_penalty = -0.01 * self.config.lambda_transport * last_action_norm_sq
        
        self.previous_probe_score = current_prob_safe
        
        # Combined intermediate reward
        return probe_reward + transport_penalty
    
    def render(self) -> np.ndarray:
        """Render the current latent as an image."""
        with torch.no_grad():
            image = self.pipe.vae.decode(
                self.current_latent / self.pipe.vae.config.scaling_factor,
                return_dict=False
            )[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        return image
    
    def get_final_image(self) -> np.ndarray:
        """Get the final generated image as numpy array."""
        return self.render()
    
    def set_intervention_window(self, start: int, end: int) -> None:
        """Set the intervention window based on layer sensitivity analysis."""
        self.config.intervention_start = start
        self.config.intervention_end = end
        print(f"Intervention window set to steps [{start}, {end}]")
    
    def set_prompts(self, prompts: List[str]) -> None:
        """Update the prompt pool for training."""
        self.prompts = prompts
        self.current_prompt_idx = 0
    
    def set_latent_encoder(self, encoder: LatentEncoder) -> None:
        """Set a pre-trained latent encoder."""
        self.latent_encoder = encoder.to(self.device)
        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.config.observation_dim,), 
            dtype=np.float32
        )
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze the latent encoder for joint training."""
        if self.latent_encoder is not None:
            for param in self.latent_encoder.parameters():
                param.requires_grad = True
    
    def get_encoder_parameters(self) -> List[torch.nn.Parameter]:
        """Get encoder parameters for optimizer."""
        if self.latent_encoder is not None:
            return list(self.latent_encoder.parameters())
        return []
    
    def get_steering_parameters(self) -> List[torch.nn.Parameter]:
        """Get steering projection parameters for optimizer."""
        return list(self.steering_proj.parameters())


# Quick test
if __name__ == "__main__":
    print("Testing DiffusionSteeringEnv...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = AetherConfig(
        model_id="rupeshs/LCM-runwayml-stable-diffusion-v1-5",  # LCM for fast inference
        num_inference_steps=8,  # LCM works well with 4-8 steps
        device=device,
        use_latent_encoder=True,  # Test with encoder
        encoded_latent_dim=256,
    )
    
    print(f"Using device: {config.device}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Observation dimension: {config.observation_dim}")
    print(f"Using latent encoder: {config.use_latent_encoder}")
    
    # Quick test without loading the full model (for CI/testing)
    if device == "cpu":
        print("\n[CPU Mode] Testing observation space without model loading...")
        env = DiffusionSteeringEnv(config, load_model=False)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Latent encoder: {env.latent_encoder is not None}")
        
        # Test encoder directly
        if env.latent_encoder is not None:
            test_input = torch.randn(1, config.latent_dim).to(device)
            test_output = env.latent_encoder(test_input)
            print(f"Encoder test: {test_input.shape} -> {test_output.shape}")
        
        # Test steering projection
        test_action = torch.randn(config.steering_dim).to(device).to(config.dtype)
        test_delta = env.steering_proj(test_action)
        print(f"Steering projection test: {test_action.shape} -> {test_delta.shape}")
        
        print("\n✓ Quick test passed (without model)!")
    else:
        print("\n[GPU Mode] Full environment test...")
        env = DiffusionSteeringEnv(config)
        obs, info = env.reset(seed=42, options={"prompt": "a cat sitting on a couch"})
        
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Initial info: {info}")
        
        # Run random policy
        total_reward = 0
        done = False
        step = 0
        while not done:
            action = env.action_space.sample() * 0.01  # Small random actions
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            print(f"Step {step}: timestep={info['timestep']:.2f}, intervened={info['intervened']}")
        
        print(f"\nEpisode complete!")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Transport cost: {info['transport_cost']:.6f}")
        
        # Save final image
        import PIL.Image
        img = env.get_final_image()
        PIL.Image.fromarray(img).save("test_output.png")
        print("Saved test_output.png")
