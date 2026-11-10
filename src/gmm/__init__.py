"""PyTorch Gaussian mixture model implementation."""

from src.gmm.em import FitResult, fit_gmm
from src.gmm.gmm import GaussianMixture, GMMConfig
from src.gmm.likelihood import average_log_likelihood, component_log_probabilities
from src.gmm.map_adaptation import (
    MAPAdaptationResult,
    SufficientStatistics,
    accumulate_sufficient_statistics,
    adapt_means_map,
)
from src.gmm.metrics import aic, bic, parameter_count

__all__ = [
    "FitResult",
    "GaussianMixture",
    "GMMConfig",
    "MAPAdaptationResult",
    "SufficientStatistics",
    "accumulate_sufficient_statistics",
    "adapt_means_map",
    "aic",
    "average_log_likelihood",
    "bic",
    "component_log_probabilities",
    "fit_gmm",
    "parameter_count",
]
