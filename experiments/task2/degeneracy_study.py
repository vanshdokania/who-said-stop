"""Demonstrate singular covariance failure and diagonal regularization."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gmm.gmm import GaussianMixture
from src.gmm.likelihood import average_log_likelihood
from src.utils.logging_utils import write_json


REGULARIZATION_EPSILON = 1e-4


def run_degeneracy_study(output_path: Path) -> dict[str, object]:
    """Create zero covariance, demonstrate Cholesky failure, and regularize it."""
    observations = torch.zeros((8, 3), dtype=torch.float64)
    singular_covariance = torch.zeros((3, 3), dtype=torch.float64)
    _, failure_info = torch.linalg.cholesky_ex(singular_covariance)
    if int(failure_info.item()) == 0:
        raise RuntimeError("The constructed covariance was not singular as expected")

    identity = torch.eye(3, dtype=torch.float64)
    regularized_covariance = (
        singular_covariance + REGULARIZATION_EPSILON * identity
    )
    factor, regularized_info = torch.linalg.cholesky_ex(regularized_covariance)
    if int(regularized_info.item()) != 0:
        raise RuntimeError("Regularized covariance still failed Cholesky factorization")
    model = GaussianMixture(
        weights=torch.ones(1, dtype=torch.float64),
        means=torch.zeros((1, 3), dtype=torch.float64),
        covariances=regularized_covariance.unsqueeze(0),
        covariance_type="full",
        reg_covar=REGULARIZATION_EPSILON,
    )
    average_ll = average_log_likelihood(observations, model)
    if not torch.isfinite(torch.tensor(average_ll)):
        raise FloatingPointError("Regularized degenerate example has non-finite LL")

    result: dict[str, object] = {
        "observations": int(observations.shape[0]),
        "dimensions": int(observations.shape[1]),
        "constructed_case": "identical observations produce a zero, singular covariance",
        "unregularized_cholesky_info": int(failure_info.item()),
        "regularization_epsilon": REGULARIZATION_EPSILON,
        "regularized_cholesky_info": int(regularized_info.item()),
        "regularized_cholesky_diagonal": factor.diagonal().tolist(),
        "regularized_average_log_likelihood": average_ll,
        "log_likelihood_is_finite": True,
    }
    write_json(output_path, result)
    print(
        f"singular Cholesky info={failure_info.item()}; "
        f"after epsilon={REGULARIZATION_EPSILON:g}, "
        f"info={regularized_info.item()}, average LL={average_ll:.6f}"
    )
    return result


def main() -> None:
    run_degeneracy_study(
        PROJECT_ROOT / "results" / "task2" / "tables" / "degeneracy_demo.json"
    )


if __name__ == "__main__":
    main()
