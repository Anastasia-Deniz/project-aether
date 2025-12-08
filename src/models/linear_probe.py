"""
Project Aether - Linear Probing & Layer Sensitivity Analysis
Phase 1: Validate that concepts are linearly separable and find optimal intervention points.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class ProbeConfig:
    """Configuration for linear probing experiments."""
    num_samples: int = 1000  # Samples per class
    num_timesteps: int = 20  # Timesteps to analyze
    test_split: float = 0.2
    random_seed: int = 42


class LinearProbe(nn.Module):
    """
    Simple linear probe for concept detection.
    Follows Alain & Bengio (2016).
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit (pre-sigmoid)."""
        return self.linear(x.flatten(start_dim=1))
    
    @classmethod
    def from_sklearn(cls, sklearn_model: LogisticRegression, input_dim: int) -> "LinearProbe":
        """Convert trained sklearn model to PyTorch module."""
        probe = cls(input_dim)
        probe.linear.weight.data = torch.from_numpy(
            sklearn_model.coef_.astype(np.float32)
        )
        probe.linear.bias.data = torch.from_numpy(
            sklearn_model.intercept_.astype(np.float32)
        )
        return probe


class LatentCollector:
    """
    Collects latent representations at multiple timesteps for probing.
    """
    
    def __init__(self, pipe, device: str = "cuda"):
        self.pipe = pipe
        self.device = device
        
    @torch.no_grad()
    def collect_trajectory(
        self,
        prompt: str,
        num_steps: int = 20,
        seed: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Run diffusion and collect latents at each timestep.
        
        Returns:
            Dict mapping timestep index -> latent tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Encode prompt
        prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        
        # Initialize from noise
        latent = torch.randn(
            1, 4, 64, 64,
            device=self.device,
            dtype=self.pipe.unet.dtype,
        )
        
        # Set scheduler
        self.pipe.scheduler.set_timesteps(num_steps, device=self.device)
        
        latents = {0: latent.clone().cpu()}  # t=1.0 (pure noise)
        
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            noise_pred = self.pipe.unet(
                latent,
                t,
                encoder_hidden_states=prompt_embeds[0],
            ).sample
            
            latent = self.pipe.scheduler.step(
                noise_pred, t, latent
            ).prev_sample
            
            latents[i + 1] = latent.clone().cpu()
        
        return latents


def collect_dataset_latents(
    pipe,
    safe_prompts: List[str],
    unsafe_prompts: List[str],
    num_timesteps: int = 20,
    device: str = "cuda",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Collect latents for safe and unsafe prompts at each timestep.
    
    Returns:
        Dict mapping timestep -> (X: latents, y: labels)
    """
    collector = LatentCollector(pipe, device)
    
    # Storage: timestep -> list of (latent, label)
    data_by_timestep = {t: [] for t in range(num_timesteps + 1)}
    
    # Collect safe samples (label=0)
    print("Collecting safe samples...")
    for i, prompt in enumerate(tqdm(safe_prompts)):
        latents = collector.collect_trajectory(prompt, num_timesteps, seed=i)
        for t, z in latents.items():
            data_by_timestep[t].append((z.flatten().numpy(), 0))
    
    # Collect unsafe samples (label=1)
    print("Collecting unsafe samples...")
    for i, prompt in enumerate(tqdm(unsafe_prompts)):
        latents = collector.collect_trajectory(prompt, num_timesteps, seed=i + len(safe_prompts))
        for t, z in latents.items():
            data_by_timestep[t].append((z.flatten().numpy(), 1))
    
    # Convert to arrays
    result = {}
    for t, data in data_by_timestep.items():
        X = np.stack([d[0] for d in data])
        y = np.array([d[1] for d in data])
        result[t] = (X, y)
    
    return result


def train_probes_per_timestep(
    data_by_timestep: Dict[int, Tuple[np.ndarray, np.ndarray]],
    test_split: float = 0.2,
    random_seed: int = 42,
) -> Dict[int, Dict]:
    """
    Train linear probes at each timestep and return results.
    
    Returns:
        Dict mapping timestep -> {
            'model': LogisticRegression,
            'train_acc': float,
            'test_acc': float,
            'auc': float,
        }
    """
    from sklearn.model_selection import train_test_split
    
    results = {}
    
    for t, (X, y) in tqdm(data_by_timestep.items(), desc="Training probes"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed, stratify=y
        )
        
        # Train logistic regression
        model = LogisticRegression(max_iter=1000, random_state=random_seed)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        results[t] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'auc': auc,
        }
    
    return results


@dataclass
class LayerSensitivityResult:
    """Result of layer sensitivity analysis for one timestep."""
    timestep: int
    probe_accuracy: float
    quality_preservation: float  # 1 - FID_normalized
    steering_effectiveness: float  # ΔSSR
    sensitivity_score: float  # Combined score


def compute_layer_sensitivity(
    probe_results: Dict[int, Dict],
    steering_effectiveness: Dict[int, float],
    quality_preservation: Dict[int, float],
) -> List[LayerSensitivityResult]:
    """
    Compute the Layer Sensitivity Score from Equation 2 in the document.
    
    S_ℓ = Acc_ℓ × (1 - FID_norm_ℓ) × ΔSSR_ℓ
    
    Args:
        probe_results: From train_probes_per_timestep
        steering_effectiveness: Dict of timestep -> ΔSSR (must be measured empirically)
        quality_preservation: Dict of timestep -> (1 - FID_norm)
    
    Returns:
        List of LayerSensitivityResult sorted by sensitivity score
    """
    results = []
    
    for t, probe_data in probe_results.items():
        acc = probe_data['test_acc']
        qual = quality_preservation.get(t, 0.5)  # Default to 0.5 if not measured
        eff = steering_effectiveness.get(t, 0.5)  # Default to 0.5 if not measured
        
        score = acc * qual * eff
        
        results.append(LayerSensitivityResult(
            timestep=t,
            probe_accuracy=acc,
            quality_preservation=qual,
            steering_effectiveness=eff,
            sensitivity_score=score,
        ))
    
    # Sort by sensitivity score (descending)
    results.sort(key=lambda x: x.sensitivity_score, reverse=True)
    
    return results


def plot_sensitivity_analysis(results: List[LayerSensitivityResult], save_path: str = None):
    """
    Recreate Figure 1 from the document: Layer Sensitivity Curves.
    """
    # Extract data
    timesteps = [r.timestep for r in results]
    # Convert to normalized time (0 = noise, 1 = image)
    max_t = max(timesteps)
    normalized_t = [1 - (t / max_t) for t in timesteps]  # Reverse so 1 = noise
    
    accuracies = [r.probe_accuracy for r in results]
    effectiveness = [r.steering_effectiveness for r in results]
    combined = [r.sensitivity_score for r in results]
    
    # Sort by normalized timestep for plotting
    sorted_data = sorted(zip(normalized_t, accuracies, effectiveness, combined))
    normalized_t, accuracies, effectiveness, combined = zip(*sorted_data)
    
    # Find optimal point
    optimal_idx = combined.index(max(combined))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(normalized_t, accuracies, 'b-', label='Probe Accuracy', linewidth=2)
    ax.plot(normalized_t, effectiveness, 'g--', label='Steering Effectiveness', linewidth=2)
    ax.plot(normalized_t, combined, 'r-', label='Combined Score $S_\\ell$', linewidth=2)
    
    # Mark optimal point
    ax.axvline(x=normalized_t[optimal_idx], color='orange', linestyle=':', 
               label=f'Optimal (t={normalized_t[optimal_idx]:.2f})', linewidth=2)
    ax.scatter([normalized_t[optimal_idx]], [combined[optimal_idx]], 
               color='orange', s=100, zorder=5)
    
    ax.set_xlabel('Timestep t (1.0 = noise, 0.0 = image)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Layer Sensitivity Analysis', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    
    return fig


# Example usage and sanity checks
if __name__ == "__main__":
    # This is a simplified example - in practice you'd use real prompts
    
    # Dummy data for testing the pipeline
    num_timesteps = 20
    latent_dim = 4 * 64 * 64
    
    print("Creating dummy data for pipeline test...")
    
    # Simulate collected data (in practice, use collect_dataset_latents)
    np.random.seed(42)
    data_by_timestep = {}
    
    for t in range(num_timesteps + 1):
        # Simulate: at early timesteps (high t), hard to separate
        # At late timesteps (low t), easy to separate
        separability = t / num_timesteps  # 0 at start, 1 at end
        
        # Safe samples: centered at 0
        X_safe = np.random.randn(100, latent_dim) * 0.5
        # Unsafe samples: shifted by separability amount
        X_unsafe = np.random.randn(100, latent_dim) * 0.5 + separability * 0.5
        
        X = np.vstack([X_safe, X_unsafe])
        y = np.array([0] * 100 + [1] * 100)
        
        data_by_timestep[t] = (X, y)
    
    print("Training probes...")
    probe_results = train_probes_per_timestep(data_by_timestep)
    
    print("\nProbe accuracies by timestep:")
    for t in sorted(probe_results.keys()):
        acc = probe_results[t]['test_acc']
        print(f"  t={t:2d}: {acc:.3f}")
    
    # Simulate steering effectiveness (would be measured empirically)
    # Effectiveness is highest in the middle
    steering_effectiveness = {}
    for t in range(num_timesteps + 1):
        # Bell curve centered at middle
        mid = num_timesteps / 2
        eff = np.exp(-((t - mid) ** 2) / (2 * (mid / 2) ** 2))
        steering_effectiveness[t] = eff
    
    # Quality preservation (inverse of FID degradation)
    # Quality drops if we intervene too early
    quality_preservation = {}
    for t in range(num_timesteps + 1):
        qual = 1 - 0.3 * (1 - t / num_timesteps)  # Better at early timesteps
        quality_preservation[t] = qual
    
    print("\nComputing layer sensitivity scores...")
    sensitivity_results = compute_layer_sensitivity(
        probe_results, steering_effectiveness, quality_preservation
    )
    
    print("\nTop 5 layers by sensitivity:")
    for r in sensitivity_results[:5]:
        print(f"  t={r.timestep:2d}: S={r.sensitivity_score:.3f} "
              f"(Acc={r.probe_accuracy:.3f}, Qual={r.quality_preservation:.3f}, "
              f"Eff={r.steering_effectiveness:.3f})")
    
    print("\nPlotting...")
    plot_sensitivity_analysis(sensitivity_results, save_path="sensitivity_analysis.png")
