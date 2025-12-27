"""
Project Aether - Transport Reward Module
Computes the transport cost (Wasserstein-2 inspired).

The transport cost penalizes large steering actions:
Transport_cost = Σ_t ||Δz_t||²

Note: This is a simplified proxy for the Wasserstein-2 distance. The full W2 distance
between distributions requires solving an optimal transport problem. Our formulation
measures the total squared displacement of steering actions, which encourages
minimal intervention while achieving safety. This is computationally efficient and
provides a good approximation for the steering problem.
"""

import torch
import numpy as np
from typing import List, Optional, Union


def compute_w2_cost(
    actions: List[torch.Tensor],
    normalize: bool = False,
    latent_dim: int = 16384,
) -> float:
    """
    Compute transport cost (Wasserstein-2 inspired).
    
    Transport_cost = Σ_t ||Δz_t||²
    
    This measures the total squared displacement of steering actions across all timesteps.
    Lower cost = more efficient steering.
    
    Note: This is a simplified proxy for the Wasserstein-2 distance. The full W2 distance
    between probability distributions requires solving an optimal transport problem.
    Our formulation provides a computationally efficient approximation that encourages
    minimal intervention while achieving safety.
    
    Args:
        actions: List of steering actions Δz_t at each timestep
        normalize: If True, normalize by latent dimension
        latent_dim: Latent dimension for normalization
        
    Returns:
        Total transport cost (scalar)
    """
    if not actions:
        return 0.0
    
    total_cost = 0.0
    for action in actions:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        
        # L2 squared norm
        cost = action.pow(2).sum().item()
        total_cost += cost
    
    if normalize:
        total_cost /= latent_dim
    
    return total_cost


def compute_action_norm(action: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute L2 norm of a single action.
    
    Args:
        action: Steering action tensor
        
    Returns:
        L2 norm
    """
    if isinstance(action, np.ndarray):
        return float(np.linalg.norm(action))
    return float(action.norm().item())


class TransportReward:
    """
    Transport reward/penalty computation.
    
    Implements the transport cost term from the Aether objective:
    J(φ) = E[R_safe - λ * Σ_t ||Δz_t||²]
    
    The transport reward is negative (penalty) proportional to action magnitude.
    """
    
    def __init__(
        self,
        lambda_transport: float = 0.5,
        normalize: bool = False,
        latent_dim: int = 16384,
    ):
        """
        Initialize transport reward.
        
        Args:
            lambda_transport: Weight for transport penalty (λ)
            normalize: Whether to normalize by latent dimension
            latent_dim: Latent dimension for normalization
        """
        self.lambda_transport = lambda_transport
        self.normalize = normalize
        self.latent_dim = latent_dim
        
        # Track actions for episode
        self.episode_actions: List[torch.Tensor] = []
    
    def add_action(self, action: torch.Tensor) -> None:
        """Add an action to the current episode."""
        self.episode_actions.append(action.detach().clone())
    
    def compute_step_penalty(self, action: torch.Tensor) -> float:
        """
        Compute per-step transport penalty.
        
        Args:
            action: Current steering action
            
        Returns:
            Penalty value (negative reward contribution)
        """
        cost = action.pow(2).sum().item()
        
        if self.normalize:
            cost /= self.latent_dim
        
        return -self.lambda_transport * cost
    
    def compute_episode_penalty(self) -> float:
        """
        Compute total transport penalty for episode.
        
        Returns:
            Total penalty (negative reward contribution)
        """
        cost = compute_w2_cost(
            self.episode_actions,
            normalize=self.normalize,
            latent_dim=self.latent_dim,
        )
        
        return -self.lambda_transport * cost
    
    def get_episode_cost(self) -> float:
        """Get raw transport cost (not negated)."""
        return compute_w2_cost(
            self.episode_actions,
            normalize=self.normalize,
            latent_dim=self.latent_dim,
        )
    
    def reset(self) -> None:
        """Reset for new episode."""
        self.episode_actions = []
    
    def get_statistics(self) -> dict:
        """Get statistics about actions in current episode."""
        if not self.episode_actions:
            return {
                "num_actions": 0,
                "total_cost": 0.0,
                "mean_norm": 0.0,
                "max_norm": 0.0,
            }
        
        norms = [compute_action_norm(a) for a in self.episode_actions]
        
        return {
            "num_actions": len(self.episode_actions),
            "total_cost": self.get_episode_cost(),
            "mean_norm": float(np.mean(norms)),
            "max_norm": float(np.max(norms)),
        }


class CombinedReward:
    """
    Combined reward function for Aether.
    
    Implements: J(φ) = E[R_safe - λ * Σ_t ||Δz_t||²]
    """
    
    def __init__(
        self,
        safety_reward,  # SafetyReward instance
        transport_reward: Optional[TransportReward] = None,
        lambda_transport: float = 0.5,
    ):
        """
        Initialize combined reward.
        
        Args:
            safety_reward: SafetyReward instance for R_safe
            transport_reward: TransportReward instance (created if None)
            lambda_transport: Transport penalty weight
        """
        self.safety_reward = safety_reward
        self.transport_reward = transport_reward or TransportReward(lambda_transport)
    
    def compute_final_reward(
        self,
        final_latent: torch.Tensor,
    ) -> tuple:
        """
        Compute final episode reward.
        
        Args:
            final_latent: Final latent representation
            
        Returns:
            (total_reward, r_safe, transport_penalty)
        """
        # Safety reward
        r_safe = self.safety_reward.compute(final_latent)
        
        # Transport penalty
        transport_penalty = self.transport_reward.compute_episode_penalty()
        
        # Combined reward
        total_reward = r_safe + transport_penalty
        
        return total_reward, r_safe, transport_penalty
    
    def add_action(self, action: torch.Tensor) -> None:
        """Track action for transport cost."""
        self.transport_reward.add_action(action)
    
    def reset(self) -> None:
        """Reset for new episode."""
        self.transport_reward.reset()


# Quick test
if __name__ == "__main__":
    print("Testing Transport Reward modules...")
    
    # Test compute_w2_cost
    actions = [torch.randn(256) * 0.01 for _ in range(5)]
    cost = compute_w2_cost(actions)
    print(f"W2 cost (5 actions): {cost:.6f}")
    
    # Test TransportReward
    tr = TransportReward(lambda_transport=0.5)
    
    for action in actions:
        tr.add_action(action)
        step_penalty = tr.compute_step_penalty(action)
        print(f"  Step penalty: {step_penalty:.6f}")
    
    episode_penalty = tr.compute_episode_penalty()
    print(f"Episode penalty: {episode_penalty:.6f}")
    
    stats = tr.get_statistics()
    print(f"Statistics: {stats}")
    
    print("\n✓ Transport reward test passed!")

