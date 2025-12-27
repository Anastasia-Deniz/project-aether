"""
Project Aether - PPO Trainer for Latent Space Steering
Phase 2: Train a policy to steer diffusion latents toward safe generations.

This implements the training loop from the Aether framework:
- PPO (Proximal Policy Optimization) for policy learning
- Optimal transport-inspired reward: J(φ) = E[R_safe - λ Σ_t ||a_t||²]
- Integration with DiffusionSteeringEnv
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.
    Default values optimized for RTX 4050 (6GB VRAM).
    """
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training (memory optimized for 6GB GPU)
    n_steps: int = 512  # Reduced from 2048 for memory
    batch_size: int = 32  # Reduced from 64 for memory
    n_epochs: int = 10
    total_timesteps: int = 100_000  # Start smaller, can increase
    
    # Policy architecture (memory optimized)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])  # Reduced
    activation: str = "relu"
    
    # Logging (more frequent due to smaller rollouts)
    log_interval: int = 5
    save_interval: int = 5000
    eval_interval: int = 2500
    
    # Device
    device: str = "cuda"


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Architecture:
    - Shared feature extractor
    - Actor head (policy): outputs mean and log_std
    - Critic head (value): outputs state value
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 256],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Activation function
        act_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }.get(activation, nn.ReLU)
        
        # Shared feature extractor
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Smaller initialization for actor output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        
        # Smaller initialization for critic output
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution params and value."""
        # Clamp observations to prevent extreme values
        obs = torch.clamp(obs, -10.0, 10.0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        features = self.shared(obs)
        
        # Clamp features for numerical stability
        features = torch.clamp(features, -10.0, 10.0)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        
        action_mean = self.actor_mean(features)
        # Clamp action_mean to prevent extreme values
        action_mean = torch.clamp(action_mean, -5.0, 5.0)
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Ensure log_std is bounded and std is always positive with minimum value
        log_std_clamped = torch.clamp(self.actor_log_std, -5.0, 2.0)  # exp(-5)≈0.007, exp(2)≈7.4
        action_std = log_std_clamped.exp().expand_as(action_mean)
        
        value = self.critic(features)
        value = torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return action_mean, action_std, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action, log_prob, value
        """
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            # Clamp sampled action
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
            # Handle NaN in log_prob
            log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-100.0)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob, value, entropy
        """
        action_mean, action_std, value = self.forward(obs)
        
        # Clamp actions for numerical stability
        actions = torch.clamp(actions, -1.0, 1.0)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Handle NaN values
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-100.0)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=10.0, neginf=0.0)
        
        return log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer to store rollout data for PPO training."""
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed during finalization
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: float):
        """Compute GAE advantages and returns."""
        # GAE (Generalized Advantage Estimation)
        last_gae_lam = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            
            delta = (
                self.rewards[t] 
                + self.gamma * next_value * next_non_terminal 
                - self.values[t]
            )
            last_gae_lam = (
                delta 
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[t] = last_gae_lam
        
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: int):
        """Generate random batches for training."""
        indices = np.random.permutation(self.ptr)
        
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield {
                'observations': torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                'actions': torch.FloatTensor(self.actions[batch_indices]).to(self.device),
                'old_log_probs': torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                'advantages': torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                'returns': torch.FloatTensor(self.returns[batch_indices]).to(self.device),
            }
    
    def reset(self):
        """Reset buffer for next rollout."""
        self.ptr = 0
        self.full = False


class AetherPPOTrainer:
    """
    PPO Trainer for the Aether framework.
    
    Trains a policy to steer diffusion latents toward safe generations
    using the optimal transport-inspired reward from Equation 7:
    
    J(φ) = E[R_safe - λ Σ_t ||a_t||²]
    
    Note: The transport cost Σ_t ||a_t||² is a simplified proxy for the Wasserstein-2
    distance. It measures the total squared displacement of steering actions, encouraging
    minimal intervention while achieving safety.
    """
    
    def __init__(
        self,
        env,
        config: PPOConfig,
        probe_path: Optional[str] = None,
        output_dir: str = "./outputs/ppo",
        experiment_name: str = "aether_ppo",
        use_wandb: bool = False,
    ):
        self.env = env
        self.config = config
        self.probe_path = probe_path  # Store probe path for later saving
        self.device = config.device
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize policy
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=config.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        
        # Load probe if provided
        if probe_path:
            self._load_probe(probe_path)
        
        # Training state
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Initialize wandb
        if self.use_wandb:
            self._init_wandb()
    
    def _load_probe(self, probe_path: str):
        """Load a pre-trained linear probe for the environment.
        
        Uses the best probe from sensitivity analysis (timestep 4) if available,
        otherwise falls back to middle timestep or any available probe.
        """
        from src.models.linear_probe import LinearProbe
        import json
        
        probe_path = Path(probe_path)
        probe_file = None
        
        if probe_path.is_dir():
            # First, try to find sensitivity scores to get the best probe
            sensitivity_file = probe_path.parent / "sensitivity_scores.json"
            best_timestep = None
            
            if sensitivity_file.exists():
                try:
                    with open(sensitivity_file, 'r') as f:
                        sensitivity_data = json.load(f)
                    
                    # Find timestep with highest score
                    best_score = -1
                    for t_str, data in sensitivity_data.items():
                        if t_str == "optimal_window":
                            continue
                        if isinstance(data, dict) and 'score' in data:
                            score = data['score']
                            if score > best_score:
                                best_score = score
                                best_timestep = int(t_str)
                    
                    if best_timestep is not None:
                        # Try to load the best probe
                        probe_file = probe_path / f"probe_t{best_timestep:02d}.pt"
                        if probe_file.exists():
                            print(f"Using best probe from sensitivity analysis: timestep {best_timestep} (score: {best_score:.3f})")
                except Exception as e:
                    print(f"Could not read sensitivity scores: {e}")
            
            # Fallback: try timestep 4 (known to be best from analysis)
            if probe_file is None or not probe_file.exists():
                probe_file = probe_path / "probe_t04.pt"
                if probe_file.exists():
                    print(f"Using probe_t04.pt (best timestep from analysis)")
            
            # Fallback: use middle timestep
            if probe_file is None or not probe_file.exists():
                probe_files = list(probe_path.glob("probe_t*.pt"))
                if probe_files:
                    probe_file = sorted(probe_files)[len(probe_files) // 2]
                    print(f"Using middle timestep probe: {probe_file.name}")
                else:
                    print(f"No probe files found in {probe_path}")
                    return
        else:
            probe_file = probe_path
        
        print(f"Loading probe: {probe_file}")
        
        probe = LinearProbe(input_dim=self.env.config.latent_dim)
        probe.load_state_dict(torch.load(probe_file, map_location=self.device))
        probe = probe.to(self.device)
        probe.eval()
        
        self.env.linear_probe = probe
        print(f"[OK] Probe loaded and set in environment")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project="project-aether",
            name=self.experiment_name,
            config={
                "ppo": vars(self.config),
                "env": {
                    "model_id": self.env.config.model_id,
                    "num_inference_steps": self.env.config.num_inference_steps,
                    "lambda_transport": self.env.config.lambda_transport,
                    "steering_dim": self.env.config.steering_dim,
                },
            },
        )
    
    def collect_rollouts(self) -> Dict[str, float]:
        """Collect rollout data using current policy."""
        self.buffer.reset()
        self.policy.eval()
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(self.config.n_steps):
            # Get action from policy - ensure observation is clean
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
            obs = np.clip(obs, -10.0, 10.0)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_tensor)
            
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            
            # Handle NaN values
            action_np = np.nan_to_num(action_np, nan=0.0)
            log_prob_np = float(np.nan_to_num(log_prob_np, nan=0.0))
            value_np = float(np.nan_to_num(value_np, nan=0.0))
            
            # Clip action to valid range
            action_np = np.clip(action_np, -1.0, 1.0)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Handle NaN reward
            reward = float(np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0))
            reward = np.clip(reward, -10.0, 10.0)  # Clip extreme rewards
            
            # Store transition
            self.buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                value=value_np,
                log_prob=log_prob_np,
                done=done,
            )
            
            episode_reward += reward
            episode_length += 1
            self.global_step += 1
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        # Compute last value for GAE
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_value = self.policy.get_action(obs_tensor)
        
        self.buffer.compute_returns_and_advantages(last_value.cpu().numpy().item())
        
        # Compute rollout stats
        return {
            'mean_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0,
        }
    
    def update_policy(self) -> Dict[str, float]:
        """Update policy using PPO."""
        self.policy.train()
        
        # Normalize advantages
        advantages = self.buffer.advantages[:self.buffer.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages[:self.buffer.ptr] = advantages
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        
        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get(self.config.batch_size):
                obs = batch['observations']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss_1 = -advantages * ratio
                policy_loss_2 = -advantages * torch.clamp(
                    ratio, 
                    1 - self.config.clip_range, 
                    1 + self.config.clip_range
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.vf_coef * value_loss 
                    + self.config.ent_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Logging metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.mean().item())
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    approx_kls.append(approx_kl)
                    clip_fractions.append(
                        ((ratio - 1).abs() > self.config.clip_range).float().mean().item()
                    )
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions),
        }
    
    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            total_timesteps: Override config total_timesteps
            
        Returns:
            Training history
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        n_updates = total_timesteps // self.config.n_steps
        
        print(f"\n{'='*60}")
        print("AETHER PPO TRAINING")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Updates: {n_updates}")
        print(f"Steps per update: {self.config.n_steps}")
        print(f"Output: {self.run_dir}")
        print(f"{'='*60}\n")
        
        history = {
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
        }
        
        progress_bar = tqdm(range(n_updates), desc="Training")
        
        for update in progress_bar:
            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            
            # MEMORY OPTIMIZATION: Clear CUDA cache after rollouts
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Update policy
            update_stats = self.update_policy()
            
            # MEMORY OPTIMIZATION: Clear CUDA cache after update
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Record history
            history['rewards'].append(rollout_stats['mean_reward'])
            history['policy_loss'].append(update_stats['policy_loss'])
            history['value_loss'].append(update_stats['value_loss'])
            history['entropy'].append(update_stats['entropy'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'reward': f"{rollout_stats['mean_reward']:.2f}",
                'policy_loss': f"{update_stats['policy_loss']:.4f}",
            })
            
            # Logging
            if update % self.config.log_interval == 0:
                self._log_metrics(update, rollout_stats, update_stats)
            
            # Save checkpoint
            if update % (self.config.save_interval // self.config.n_steps) == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
        
        # Final save
        self.save_checkpoint("final_policy.pt")
        self._save_history(history)
        
        print(f"\n{'='*60}", flush=True)
        print("[SUCCESS] TRAINING COMPLETE", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Final mean reward: {np.mean(history['rewards'][-10:]):.4f}")
        print(f"Output: {self.run_dir}")
        
        return history
    
    def _log_metrics(
        self, 
        update: int, 
        rollout_stats: Dict, 
        update_stats: Dict
    ):
        """Log metrics to console and wandb."""
        if self.use_wandb:
            wandb.log({
                'rollout/mean_reward': rollout_stats['mean_reward'],
                'rollout/mean_length': rollout_stats['mean_length'],
                'train/policy_loss': update_stats['policy_loss'],
                'train/value_loss': update_stats['value_loss'],
                'train/entropy': update_stats['entropy'],
                'train/approx_kl': update_stats['approx_kl'],
                'train/clip_fraction': update_stats['clip_fraction'],
                'global_step': self.global_step,
            })
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': vars(self.config),
        }
        torch.save(checkpoint, self.run_dir / filename)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
    
    def _save_history(self, history: Dict):
        """Save training history to JSON."""
        # Convert numpy arrays to lists
        history_serializable = {
            k: [float(v) for v in vals]
            for k, vals in history.items()
        }

        with open(self.run_dir / "training_history.json", 'w') as f:
            json.dump(history_serializable, f, indent=2)

        # Save training configuration and probe info
        training_config = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'probe_path': self.probe_path,
            'probe_run': Path(self.probe_path).parent.name if self.probe_path else None,
            'ppo_config': {
                'learning_rate': self.config.learning_rate,
                'gamma': self.config.gamma,
                'gae_lambda': self.config.gae_lambda,
                'clip_range': self.config.clip_range,
                'vf_coef': self.config.vf_coef,
                'ent_coef': self.config.ent_coef,
                'max_grad_norm': self.config.max_grad_norm,
                'n_steps': self.config.n_steps,
                'batch_size': self.config.batch_size,
                'n_epochs': self.config.n_epochs,
                'total_timesteps': self.config.total_timesteps,
            },
            'env_config': {
                'model_id': self.env.config.model_id if hasattr(self.env, 'config') else None,
                'num_inference_steps': self.env.config.num_inference_steps if hasattr(self.env, 'config') else None,
                'guidance_scale': self.env.config.guidance_scale if hasattr(self.env, 'config') else None,
                'intervention_start': self.env.config.intervention_start if hasattr(self.env, 'config') else None,
                'intervention_end': self.env.config.intervention_end if hasattr(self.env, 'config') else None,
                'lambda_transport': self.env.config.lambda_transport if hasattr(self.env, 'config') else None,
            }
        }

        with open(self.run_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)


# Quick test
if __name__ == "__main__":
    print("Testing PPO components...")
    
    # Test ActorCritic
    obs_dim = 258  # encoded_latent (256) + t (1) + score (1)
    action_dim = 256  # steering_dim
    
    policy = ActorCritic(obs_dim, action_dim)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Test forward pass
    obs = torch.randn(4, obs_dim)
    action, log_prob, value = policy.get_action(obs)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test evaluate
    log_prob, value, entropy = policy.evaluate_actions(obs, action)
    print(f"Entropy: {entropy.mean().item():.4f}")
    
    print("\n✓ PPO components test passed!")

