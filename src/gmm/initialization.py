"""Random and PyTorch k-means initializers for Gaussian mixtures."""

import torch

from src.gmm.gmm import CovarianceType, GaussianMixture, InitializationType


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _global_covariance(data: torch.Tensor, reg_covar: float) -> torch.Tensor:
    centered = data - data.mean(dim=0)
    covariance = centered.T @ centered / max(data.shape[0], 1)
    covariance.diagonal().add_(reg_covar)
    return covariance


def _pack_covariances(
    covariance: torch.Tensor,
    n_components: int,
    covariance_type: CovarianceType,
) -> torch.Tensor:
    if covariance_type == "full":
        return covariance.expand(n_components, -1, -1).clone()
    if covariance_type == "diagonal":
        return covariance.diagonal().expand(n_components, -1).clone()
    if covariance_type == "tied":
        return covariance.clone()
    return covariance.diagonal().mean().expand(n_components).clone()


def random_initialization(
    data: torch.Tensor,
    n_components: int,
    covariance_type: CovarianceType,
    reg_covar: float,
    seed: int,
) -> GaussianMixture:
    """Choose distinct data points as means and use a regularized global covariance."""
    generator = _generator(seed, data.device)
    indices = torch.randperm(data.shape[0], generator=generator, device=data.device)
    means = data[indices[:n_components]].clone()
    covariance = _global_covariance(data, reg_covar)
    weights = torch.full(
        (n_components,),
        1.0 / n_components,
        dtype=data.dtype,
        device=data.device,
    )
    return GaussianMixture(
        weights,
        means,
        _pack_covariances(covariance, n_components, covariance_type),
        covariance_type,
        reg_covar,
    )


def _kmeans(
    data: torch.Tensor,
    n_components: int,
    seed: int,
    max_iter: int = 30,
) -> torch.Tensor:
    """Run Lloyd's algorithm using only PyTorch tensor operations."""
    generator = _generator(seed, data.device)
    indices = torch.randperm(data.shape[0], generator=generator, device=data.device)
    centers = data[indices[:n_components]].clone()

    for _ in range(max_iter):
        labels = torch.cdist(data, centers).argmin(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, data)
        counts = torch.bincount(labels, minlength=n_components)
        updated = sums / counts.clamp_min(1).to(data.dtype).unsqueeze(1)
        empty = counts == 0
        if empty.any():
            replacement = torch.randperm(
                data.shape[0],
                generator=generator,
                device=data.device,
            )[: int(empty.sum().item())]
            updated[empty] = data[replacement]
        if torch.equal(updated, centers):
            break
        centers = updated
    return centers


def kmeans_initialization(
    data: torch.Tensor,
    n_components: int,
    covariance_type: CovarianceType,
    reg_covar: float,
    seed: int,
) -> GaussianMixture:
    """Initialize means with PyTorch k-means and covariances from global data."""
    means = _kmeans(data, n_components, seed)
    covariance = _global_covariance(data, reg_covar)
    weights = torch.full(
        (n_components,),
        1.0 / n_components,
        dtype=data.dtype,
        device=data.device,
    )
    return GaussianMixture(
        weights,
        means,
        _pack_covariances(covariance, n_components, covariance_type),
        covariance_type,
        reg_covar,
    )


def initialize_gmm(
    data: torch.Tensor,
    n_components: int,
    covariance_type: CovarianceType,
    reg_covar: float,
    seed: int,
    method: InitializationType,
) -> GaussianMixture:
    """Create model parameters using the requested deterministic method."""
    if n_components > data.shape[0]:
        raise ValueError("n_components cannot exceed the number of observations")
    if method == "random":
        return random_initialization(
            data, n_components, covariance_type, reg_covar, seed
        )
    if method == "kmeans":
        return kmeans_initialization(
            data, n_components, covariance_type, reg_covar, seed
        )
    raise ValueError(f"Unsupported initialization method: {method}")
