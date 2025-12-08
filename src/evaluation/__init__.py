"""
Project Aether - Evaluation Module
Contains metrics and benchmark utilities for Phase 3.
"""

from .metrics import (
    compute_ssr,
    compute_lpips,
    compute_transport_cost,
    compute_fpr,
    EvaluationMetrics,
)

__all__ = [
    'compute_ssr',
    'compute_lpips', 
    'compute_transport_cost',
    'compute_fpr',
    'EvaluationMetrics',
]

