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
    # Using v1.5 as it doesn't require HuggingFace authentication
    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 15  # Reduced from 20 for 6GB GPUs
    guidance_scale: float = 7.5
    latent_channels: int = 4
    latent_size: int = 64
    steering_dim: int = 256  # Low-rank steering subspace
    lambda_transport: float = 0.5  # λ in the reward
    device: str = "cuda"
    dtype: torch.dtype = torch.float16  # CRITICAL: Half precision for memory
    
    # Intervention window (adjusted for 15 steps)
    # Update after running Phase 1 sensitivity analysis
    intervention_start: int = 6   # ~40% of generation
    intervention_end: int = 11    # ~73% of generation
    
    # Action constraints
    max_action_norm: float = 0.1  # Clip steering magnitude
    
    # Latent encoder settings (reduces observation from 16K to 256)
    use_latent_encoder: bool = True
    encoded_latent_dim: int = 256
    encoder_hidden_dim: int = 512  # Reduced from 1024 for memory
    
    # Steering projection initialization
    steering_init: str = "orthogonal"  # "orthogonal", "random", "learned"
    
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
        self.actions_taken = []
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to a new episode with a prompt."""
        super().reset(seed=seed)
        
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
        self.trajectory = [self.current_latent.clone()]
        self.actions_taken = []
        
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
        
        # Track trajectory
        self.trajectory.append(self.current_latent.clone())
        if should_intervene:
            self.actions_taken.append(delta_z.clone())
        
        # Advance step counter
        self.current_step += 1
        terminated = self.current_step >= self.config.num_inference_steps
        
        # Compute reward (only at terminal state for efficiency)
        if terminated:
            reward = self._compute_final_reward()
        else:
            reward = 0.0  # Intermediate reward is 0, final reward is all that matters
        
        # Get observation
        obs = self._get_observation()
        
        # Compute transport cost so far
        transport_cost = sum(
            a.pow(2).sum().item() for a in self.actions_taken
        ) if self.actions_taken else 0.0
        
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
        
        # Encode latent if encoder is available (reduces from 16K to 256 dims)
        if self.latent_encoder is not None:
            with torch.no_grad():
                z_encoded = self.latent_encoder(z_flat.unsqueeze(0).to(self.device))
                z_obs = z_encoded.squeeze(0).cpu().numpy()
        else:
            z_obs = z_flat.cpu().numpy()
        
        # Normalized timestep
        t = np.array([1.0 - (self.current_step / self.config.num_inference_steps)])
        
        # Safety score from linear probe
        if self.linear_probe is not None:
            with torch.no_grad():
                score = self.linear_probe(z_flat.unsqueeze(0).to(self.device))
                score = torch.sigmoid(score).cpu().numpy().flatten()
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
        if self.actions_taken:
            transport_cost = sum(
                a.pow(2).sum().item() for a in self.actions_taken
            )
        else:
            transport_cost = 0.0
        
        # Final reward: R_safe - λ * W2_cost
        reward = r_safe - self.config.lambda_transport * transport_cost
        
        return reward
    
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
        model_id="stabilityai/stable-diffusion-2-1-base",
        num_inference_steps=10,  # Very fast for testing
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
