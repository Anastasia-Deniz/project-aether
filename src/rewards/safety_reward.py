"""
Project Aether - Safety Reward Module
Computes R_safe from the Aether framework.

The safety reward measures how "safe" a generated image is:
- R_safe = +1 for safe content
- R_safe = -1 for unsafe content
- R_safe ∈ [-1, 1] for probabilistic classifiers
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from abc import ABC, abstractmethod


class SafetyReward(ABC):
    """Abstract base class for safety reward computation."""
    
    @abstractmethod
    def compute(self, x: torch.Tensor) -> float:
        """
        Compute safety reward for input.
        
        Args:
            x: Input tensor (image or latent)
            
        Returns:
            Safety reward in [-1, 1] where +1 is safe, -1 is unsafe
        """
        pass


class ProbeBasedSafetyReward(SafetyReward):
    """
    Safety reward using a linear probe on latent space.
    
    This is the default reward for the Aether framework, using
    the probes trained in Phase 1.
    """
    
    def __init__(
        self,
        probe: nn.Module,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        """
        Initialize probe-based safety reward.
        
        Args:
            probe: Trained linear probe (from Phase 1)
            device: Computation device
            threshold: Classification threshold
        """
        self.probe = probe.to(device)
        self.probe.eval()
        self.device = device
        self.threshold = threshold
        
        # Freeze probe
        for param in self.probe.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def compute(self, z: torch.Tensor) -> float:
        """
        Compute safety reward from latent.
        
        The probe predicts P(unsafe | z). We convert this to a reward:
        R_safe = 1 - 2 * P(unsafe | z)
        
        This maps:
        - P(unsafe) = 0 → R = +1 (safe)
        - P(unsafe) = 1 → R = -1 (unsafe)
        - P(unsafe) = 0.5 → R = 0 (neutral)
        
        Args:
            z: Latent tensor (flattened or 4D)
            
        Returns:
            Safety reward in [-1, 1]
        """
        # Flatten if needed
        if z.dim() == 4:
            z = z.flatten(start_dim=1)
        elif z.dim() == 1:
            z = z.unsqueeze(0)
        
        z = z.float().to(self.device)
        
        # Get probe prediction
        logit = self.probe(z)
        prob_unsafe = torch.sigmoid(logit).item()
        
        # Convert to reward: safe → +1, unsafe → -1
        reward = 1.0 - 2.0 * prob_unsafe
        
        return reward
    
    @torch.no_grad()
    def get_probability(self, z: torch.Tensor) -> float:
        """Get raw probability of unsafe content."""
        if z.dim() == 4:
            z = z.flatten(start_dim=1)
        elif z.dim() == 1:
            z = z.unsqueeze(0)
        
        z = z.float().to(self.device)
        logit = self.probe(z)
        return torch.sigmoid(logit).item()
    
    @classmethod
    def from_checkpoint(
        cls,
        probe_path: str,
        latent_dim: int = 16384,
        device: str = "cuda",
    ) -> "ProbeBasedSafetyReward":
        """
        Load safety reward from probe checkpoint.
        
        Args:
            probe_path: Path to probe .pt file
            latent_dim: Latent dimension (default: 64*64*4)
            device: Computation device
            
        Returns:
            ProbeBasedSafetyReward instance
        """
        from src.models.linear_probe import LinearProbe
        
        probe = LinearProbe(input_dim=latent_dim)
        probe.load_state_dict(torch.load(probe_path, map_location=device))
        
        return cls(probe, device=device)


class ClassifierBasedSafetyReward(SafetyReward):
    """
    Safety reward using an image-space classifier.
    
    This is an alternative to probe-based rewards, using a 
    pre-trained NSFW classifier on the decoded image.
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        device: str = "cuda",
    ):
        """
        Initialize classifier-based safety reward.
        
        Args:
            classifier: Image classifier returning safety scores
            device: Computation device
        """
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.device = device
        
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def compute(self, image: torch.Tensor) -> float:
        """
        Compute safety reward from image.
        
        Args:
            image: Image tensor (N, C, H, W) in [0, 1]
            
        Returns:
            Safety reward in [-1, 1]
        """
        image = image.to(self.device)
        score = self.classifier(image)
        
        # Assume classifier outputs safety score in [0, 1]
        # where 1 = safe, 0 = unsafe
        if score.dim() > 0:
            score = score.mean()
        
        # Convert to [-1, 1] range
        reward = 2.0 * score.item() - 1.0
        
        return reward


# Quick test
if __name__ == "__main__":
    print("Testing Safety Reward modules...")
    
    # Test with dummy probe
    class DummyProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16384, 1)
        
        def forward(self, x):
            return self.linear(x.flatten(start_dim=1))
    
    probe = DummyProbe()
    reward_fn = ProbeBasedSafetyReward(probe, device="cpu")
    
    # Test with random latent
    z = torch.randn(1, 4, 64, 64)
    reward = reward_fn.compute(z)
    print(f"Random latent reward: {reward:.4f}")
    
    prob = reward_fn.get_probability(z)
    print(f"P(unsafe): {prob:.4f}")
    
    print("\n✓ Safety reward test passed!")

