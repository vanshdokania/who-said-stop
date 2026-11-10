"""Mean-only MAP adaptation of a fixed Gaussian mixture model."""

from dataclasses import dataclass

import torch

from src.gmm.gmm import GaussianMixture
from src.gmm.likelihood import component_log_probabilities


@dataclass(frozen=True)
class SufficientStatistics:
    """Zero-order occupancies and first-order feature sums."""

    occupancies: torch.Tensor
    first_order: torch.Tensor


@dataclass(frozen=True)
class MAPAdaptationResult:
    """Adapted speaker means and statistics for a fixed UBM."""

    model: GaussianMixture
    statistics: SufficientStatistics
    retained_components: int


def accumulate_sufficient_statistics(
    observations: torch.Tensor,
    ubm: GaussianMixture,
    chunk_size: int = 1024,
) -> SufficientStatistics:
    """Accumulate UBM-posterior zero- and first-order statistics in Torch."""
    if observations.ndim != 2 or observations.shape[1] != ubm.n_features:
        raise ValueError(
            f"Expected observations shaped [N, {ubm.n_features}], "
            f"got {tuple(observations.shape)}"
        )
    if observations.shape[0] == 0:
        raise ValueError("Cannot adapt from an empty set of observations")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not observations.is_floating_point():
        raise TypeError("observations must have a floating-point dtype")
    if not torch.isfinite(observations).all():
        raise ValueError("observations contain NaN or infinite values")

    data = observations.detach().to(
        dtype=ubm.means.dtype,
        device=ubm.means.device,
    )
    occupancies = torch.zeros(
        ubm.n_components,
        dtype=ubm.means.dtype,
        device=ubm.means.device,
    )
    first_order = torch.zeros_like(ubm.means)
    for start in range(0, data.shape[0], chunk_size):
        batch = data[start : start + chunk_size]
        weighted_log_probabilities = component_log_probabilities(
            batch,
            ubm,
            chunk_size,
        )
        log_normalizer = torch.logsumexp(
            weighted_log_probabilities,
            dim=1,
            keepdim=True,
        )
        responsibilities = (weighted_log_probabilities - log_normalizer).exp()
        occupancies += responsibilities.sum(dim=0)
        first_order += responsibilities.T @ batch

    if not torch.isfinite(occupancies).all() or not torch.isfinite(first_order).all():
        raise FloatingPointError("MAP sufficient statistics are non-finite")
    return SufficientStatistics(occupancies, first_order)


def adapt_means_map(
    observations: torch.Tensor,
    ubm: GaussianMixture,
    relevance_factor: float,
    occupancy_epsilon: float = 1e-8,
    chunk_size: int = 1024,
) -> MAPAdaptationResult:
    """Adapt only GMM means, preserving UBM weights and covariance parameters."""
    if relevance_factor <= 0:
        raise ValueError("relevance_factor must be positive")
    if occupancy_epsilon < 0:
        raise ValueError("occupancy_epsilon cannot be negative")
    if ubm.covariance_type not in ("diagonal", "full", "tied", "spherical"):
        raise ValueError(f"Unsupported UBM covariance type: {ubm.covariance_type}")

    statistics = accumulate_sufficient_statistics(observations, ubm, chunk_size)
    active = statistics.occupancies > occupancy_epsilon
    adapted_means = ubm.means.clone()
    if active.any():
        adapted_means[active] = (
            statistics.first_order[active]
            + relevance_factor * ubm.means[active]
        ) / (
            statistics.occupancies[active, None] + relevance_factor
        )
    if not torch.isfinite(adapted_means).all():
        raise FloatingPointError("MAP adaptation produced non-finite means")

    adapted_model = GaussianMixture(
        weights=ubm.weights.clone(),
        means=adapted_means,
        covariances=ubm.covariances.clone(),
        covariance_type=ubm.covariance_type,
        reg_covar=ubm.reg_covar,
    )
    return MAPAdaptationResult(
        model=adapted_model,
        statistics=statistics,
        retained_components=int((~active).sum().item()),
    )
