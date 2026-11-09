"""Data structures for Gaussian mixture models."""

from dataclasses import dataclass
from typing import Literal

import torch


CovarianceType = Literal["full", "diagonal", "tied", "spherical"]
InitializationType = Literal["random", "kmeans"]
SUPPORTED_COVARIANCE_TYPES: tuple[CovarianceType, ...] = (
    "full",
    "diagonal",
    "tied",
    "spherical",
)


@dataclass(frozen=True)
class GMMConfig:
    """Numerical and convergence settings shared by GMM experiments."""

    n_components: int
    covariance_type: CovarianceType
    reg_covar: float = 1e-4
    max_iter: int = 40
    tol: float = 1e-4
    chunk_size: int = 1024
    min_component_weight: float = 1e-8

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be at least one")
        if self.covariance_type not in SUPPORTED_COVARIANCE_TYPES:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")
        if self.reg_covar <= 0:
            raise ValueError("reg_covar must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least one")
        if self.tol < 0:
            raise ValueError("tol cannot be negative")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least one")
        if not 0 < self.min_component_weight < 1:
            raise ValueError("min_component_weight must be between zero and one")


@dataclass(frozen=True)
class GaussianMixture:
    """Fitted mixture weights, means, and covariance parameters."""

    weights: torch.Tensor
    means: torch.Tensor
    covariances: torch.Tensor
    covariance_type: CovarianceType
    reg_covar: float

    @property
    def n_components(self) -> int:
        return int(self.weights.numel())

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])
