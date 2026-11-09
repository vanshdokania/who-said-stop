"""Information criteria for fitted Gaussian mixture models."""

import math

import torch

from src.gmm.gmm import GaussianMixture
from src.gmm.likelihood import total_log_likelihood


def parameter_count(model: GaussianMixture) -> int:
    """Count independently fitted mixture weights, means, and covariances."""
    components = model.n_components
    dimensions = model.n_features
    weight_parameters = components - 1
    mean_parameters = components * dimensions
    triangular = dimensions * (dimensions + 1) // 2
    if model.covariance_type == "full":
        covariance_parameters = components * triangular
    elif model.covariance_type == "diagonal":
        covariance_parameters = components * dimensions
    elif model.covariance_type == "tied":
        covariance_parameters = triangular
    else:
        covariance_parameters = components
    return weight_parameters + mean_parameters + covariance_parameters


def aic(model: GaussianMixture, log_likelihood: float) -> float:
    """Calculate AIC from the model parameter count and total log likelihood."""
    return 2 * parameter_count(model) - 2 * log_likelihood


def bic(model: GaussianMixture, log_likelihood: float, n_observations: int) -> float:
    """Calculate BIC using the fitting observation count."""
    if n_observations < 1:
        raise ValueError("n_observations must be positive")
    return parameter_count(model) * math.log(n_observations) - 2 * log_likelihood


def information_criteria(
    model: GaussianMixture,
    observations: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[float, float, float]:
    """Return fitting log likelihood, AIC, and BIC for the supplied data."""
    likelihood = total_log_likelihood(observations, model, chunk_size)
    return (
        likelihood,
        aic(model, likelihood),
        bic(model, likelihood, observations.shape[0]),
    )
