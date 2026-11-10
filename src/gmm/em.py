"""Expectation-maximization fitting for Gaussian mixture models."""

from dataclasses import dataclass

import torch

from src.gmm.gmm import GMMConfig, GaussianMixture, InitializationType
from src.gmm.initialization import initialize_gmm
from src.gmm.likelihood import component_log_probabilities


@dataclass(frozen=True)
class FitResult:
    """Model and convergence details returned by an EM run."""

    model: GaussianMixture
    log_likelihood: float
    iterations: int
    converged: bool
    likelihood_history: tuple[float, ...]
    initialization: InitializationType
    seed: int


def _log_responsibilities(
    data: torch.Tensor,
    model: GaussianMixture,
    chunk_size: int,
) -> tuple[torch.Tensor, float]:
    scores = component_log_probabilities(data, model, chunk_size)
    normalizers = torch.logsumexp(scores, dim=1, keepdim=True)
    return (scores - normalizers).exp(), float(normalizers.sum().item())


def _regularized_covariance(
    covariance: torch.Tensor,
    reg_covar: float,
) -> torch.Tensor:
    result = covariance.clone()
    result.diagonal().add_(reg_covar)
    return result


def _maximization_step(
    data: torch.Tensor,
    responsibilities: torch.Tensor,
    model: GaussianMixture,
    config: GMMConfig,
    seed: int,
) -> GaussianMixture:
    component_count = model.n_components
    feature_count = model.n_features
    masses = responsibilities.sum(dim=0)
    safe_masses = masses.clamp_min(torch.finfo(data.dtype).eps)
    weighted_sum = responsibilities.T @ data
    means = weighted_sum / safe_masses.unsqueeze(1)
    low_mass = masses < max(data.shape[0] * config.min_component_weight, 1e-8)

    generator = torch.Generator(device=data.device)
    generator.manual_seed(seed)
    if low_mass.any():
        replacements = torch.randperm(
            data.shape[0],
            generator=generator,
            device=data.device,
        )[: int(low_mass.sum().item())]
        means[low_mass] = data[replacements]
        masses[low_mass] = max(data.shape[0] * config.min_component_weight, 1e-8)

    weights = masses.clamp_min(config.min_component_weight)
    weights = weights / weights.sum()
    covariances = []

    if config.covariance_type == "full":
        for component in range(component_count):
            difference = data - means[component]
            weighted_difference = difference * responsibilities[:, component : component + 1]
            scatter = weighted_difference.T @ difference
            covariances.append(scatter)
        covariance_tensor = torch.stack(
            [
                _regularized_covariance(
                    scatter / safe_masses[component],
                    config.reg_covar,
                )
                for component, scatter in enumerate(covariances)
            ]
        )
    elif config.covariance_type == "tied":
        tied_scatter = data.T @ data - (means.T * safe_masses) @ means
        tied = tied_scatter / safe_masses.sum()
        covariance_tensor = _regularized_covariance(tied, config.reg_covar)
    elif config.covariance_type == "diagonal":
        weighted_squared_sum = responsibilities.T @ data.square()
        variances = (
            weighted_squared_sum / safe_masses.unsqueeze(1) - means.square()
        ).clamp_min(config.reg_covar)
        covariance_tensor = variances
    else:
        weighted_squared_sum = responsibilities.T @ data.square()
        variances = (
            weighted_squared_sum / safe_masses.unsqueeze(1) - means.square()
        ).clamp_min(config.reg_covar)
        covariance_tensor = (
            variances.mean(dim=1).clamp_min(config.reg_covar)
        )

    if not torch.isfinite(means).all() or not torch.isfinite(covariance_tensor).all():
        raise FloatingPointError("EM update produced non-finite parameters")
    return GaussianMixture(
        weights=weights,
        means=means,
        covariances=covariance_tensor,
        covariance_type=config.covariance_type,
        reg_covar=config.reg_covar,
    )


def fit_gmm(
    observations: torch.Tensor,
    config: GMMConfig,
    seed: int,
    initialization: InitializationType = "random",
) -> FitResult:
    """Fit one GMM using deterministic EM and return its final training likelihood."""
    if observations.ndim != 2:
        raise ValueError("observations must have shape [N, D]")
    if observations.shape[0] < config.n_components:
        raise ValueError("there must be at least one observation per component")
    if observations.shape[1] < 1:
        raise ValueError("observations must have at least one feature")
    if not observations.is_floating_point():
        raise TypeError("observations must have a floating-point dtype")
    if not torch.isfinite(observations).all():
        raise ValueError("observations contain NaN or infinite values")

    data = observations.detach().to(dtype=torch.float64, device="cpu").contiguous()
    model = initialize_gmm(
        data,
        config.n_components,
        config.covariance_type,
        config.reg_covar,
        seed,
        initialization,
    )
    history = []
    previous_likelihood = float("-inf")
    converged = False

    for iteration in range(1, config.max_iter + 1):
        responsibilities, current_likelihood = _log_responsibilities(
            data,
            model,
            config.chunk_size,
        )
        history.append(current_likelihood)
        model = _maximization_step(
            data,
            responsibilities,
            model,
            config,
            seed + iteration,
        )
        if previous_likelihood != float("-inf"):
            change = abs(current_likelihood - previous_likelihood)
            if change <= config.tol * (1.0 + abs(previous_likelihood)):
                converged = True
                break
        previous_likelihood = current_likelihood

    _, final_likelihood = _log_responsibilities(data, model, config.chunk_size)
    if not torch.isfinite(torch.tensor(final_likelihood)):
        raise FloatingPointError("Final GMM log likelihood is not finite")
    if not history or final_likelihood != history[-1]:
        history.append(final_likelihood)
    return FitResult(
        model=model,
        log_likelihood=final_likelihood,
        iterations=iteration,
        converged=converged,
        likelihood_history=tuple(history),
        initialization=initialization,
        seed=seed,
    )
