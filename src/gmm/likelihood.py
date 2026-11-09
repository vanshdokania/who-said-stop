"""Stable log-domain probability calculations for Gaussian mixtures."""

import math

import torch

from src.gmm.gmm import GaussianMixture


LOG_2PI = math.log(2.0 * math.pi)


def _full_covariances(model: GaussianMixture) -> torch.Tensor:
    if model.covariance_type == "full":
        return model.covariances
    if model.covariance_type == "tied":
        return model.covariances.unsqueeze(0).expand(model.n_components, -1, -1)
    if model.covariance_type == "diagonal":
        return torch.diag_embed(model.covariances)
    diagonal = model.covariances[:, None].expand(-1, model.n_features)
    return torch.diag_embed(diagonal)


def _cholesky_factors(model: GaussianMixture) -> tuple[torch.Tensor, torch.Tensor]:
    covariances = _full_covariances(model)
    identity = torch.eye(
        model.n_features,
        dtype=model.means.dtype,
        device=model.means.device,
    )
    factors = []
    log_determinants = []
    for covariance in covariances:
        jitter = 0.0
        for attempt in range(7):
            candidate = covariance if jitter == 0 else covariance + jitter * identity
            factor, info = torch.linalg.cholesky_ex(candidate)
            if int(info.item()) == 0:
                factors.append(factor)
                log_determinants.append(2 * torch.log(factor.diagonal()).sum())
                break
            jitter = model.reg_covar * (10.0**attempt)
        else:
            raise RuntimeError(
                "Covariance remained non-positive-definite after regularization"
            )
    return torch.stack(factors), torch.stack(log_determinants)


def component_log_probabilities(
    data: torch.Tensor,
    model: GaussianMixture,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Return weighted component log probabilities with bounded batch memory."""
    if data.ndim != 2 or data.shape[1] != model.n_features:
        raise ValueError(
            f"Expected observations shaped [N, {model.n_features}], "
            f"got {tuple(data.shape)}"
        )
    if not torch.isfinite(data).all():
        raise ValueError("data contains NaN or infinite values")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")
    data = data.to(dtype=model.means.dtype, device=model.means.device)
    factors, log_determinants = _cholesky_factors(model)
    constant = model.n_features * LOG_2PI
    log_weights = model.weights.clamp_min(torch.finfo(model.weights.dtype).tiny).log()
    output = []

    for start in range(0, data.shape[0], chunk_size):
        batch = data[start : start + chunk_size]
        component_scores = []
        for component in range(model.n_components):
            difference = batch - model.means[component]
            whitened = torch.linalg.solve_triangular(
                factors[component],
                difference.T,
                upper=False,
            )
            mahalanobis = whitened.square().sum(dim=0)
            component_scores.append(
                log_weights[component]
                - 0.5 * (constant + log_determinants[component] + mahalanobis)
            )
        output.append(torch.stack(component_scores, dim=1))
    return torch.cat(output, dim=0)


def average_log_likelihood(
    data: torch.Tensor,
    model: GaussianMixture,
    chunk_size: int = 1024,
) -> float:
    """Calculate average per-observation mixture log likelihood."""
    if data.shape[0] == 0:
        raise ValueError("Cannot calculate likelihood for an empty dataset")
    component_scores = component_log_probabilities(data, model, chunk_size)
    return float(torch.logsumexp(component_scores, dim=1).mean().item())


def total_log_likelihood(
    data: torch.Tensor,
    model: GaussianMixture,
    chunk_size: int = 1024,
) -> float:
    """Calculate total mixture log likelihood for a set of observations."""
    if data.shape[0] == 0:
        raise ValueError("Cannot calculate likelihood for an empty dataset")
    component_scores = component_log_probabilities(data, model, chunk_size)
    return float(torch.logsumexp(component_scores, dim=1).sum().item())
