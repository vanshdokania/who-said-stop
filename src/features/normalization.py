"""Normalization for phase-derived feature vectors."""

import torch


def normalize_feature_vectors(features: torch.Tensor) -> torch.Tensor:
    """L2-normalize each frame while preserving all-zero vectors."""
    if features.ndim != 2:
        raise ValueError(f"Expected [num_frames, feature_dimension], got {features.ndim} dimensions")
    if not features.is_floating_point():
        raise TypeError("features must have a floating-point dtype")
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values")
    if features.shape[0] == 0:
        return features

    norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)
    return torch.where(norms > 0, features / norms.clamp_min(torch.finfo(features.dtype).eps), 0.0)
