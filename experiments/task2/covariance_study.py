"""Compare GMM covariance structures and component counts."""

import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gmm import (
    GMMConfig,
    aic,
    average_log_likelihood,
    bic,
    fit_gmm,
    parameter_count,
)
from src.utils.logging_utils import write_json
from experiments.task2.prepare_split import prepare_background_observations


COVARIANCE_TYPES = ("full", "diagonal", "tied", "spherical")
COMPONENT_COUNTS = (1, 2, 4, 8)
FIT_SEED = 3200
MAX_ITERATIONS = 40
REG_COVAR = 1e-4


def run_covariance_study(
    fitting_data: torch.Tensor,
    heldout_data: torch.Tensor,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Fit the requested grid and save fitting and held-out criteria."""
    rows = []
    for covariance_type in COVARIANCE_TYPES:
        for n_components in COMPONENT_COUNTS:
            seed = FIT_SEED + 100 * COVARIANCE_TYPES.index(covariance_type) + n_components
            config = GMMConfig(
                n_components=n_components,
                covariance_type=covariance_type,
                reg_covar=REG_COVAR,
                max_iter=MAX_ITERATIONS,
            )
            result = fit_gmm(
                fitting_data,
                config,
                seed=seed,
                initialization="random",
            )
            heldout_average = average_log_likelihood(
                heldout_data,
                result.model,
                config.chunk_size,
            )
            rows.append(
                {
                    "covariance_type": covariance_type,
                    "n_components": n_components,
                    "n_parameters": parameter_count(result.model),
                    "fitting_frames": fitting_data.shape[0],
                    "heldout_frames": heldout_data.shape[0],
                    "train_log_likelihood": result.log_likelihood,
                    "train_average_log_likelihood": (
                        result.log_likelihood / fitting_data.shape[0]
                    ),
                    "aic": aic(result.model, result.log_likelihood),
                    "bic": bic(
                        result.model,
                        result.log_likelihood,
                        fitting_data.shape[0],
                    ),
                    "heldout_average_log_likelihood": heldout_average,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "seed": seed,
                    "reg_covar": REG_COVAR,
                }
            )
            print(
                f"{covariance_type:9s} K={n_components}: "
                f"heldout avg LL={heldout_average:.6f}, "
                f"BIC={rows[-1]['bic']:.3f}, "
                f"iterations={result.iterations}, converged={result.converged}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "covariance_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "covariance_comparison.json",
        {
            "covariance_types": list(COVARIANCE_TYPES),
            "component_counts": list(COMPONENT_COUNTS),
            "initialization": "random",
            "fit_seed_base": FIT_SEED,
            "reg_covar": REG_COVAR,
            "max_iterations": MAX_ITERATIONS,
            "results": rows,
        },
    )
    return rows


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / "task2"
    fitting_data, heldout_data, _ = prepare_background_observations(
        PROJECT_ROOT,
        results_dir / "tables",
    )
    run_covariance_study(
        fitting_data,
        heldout_data,
        results_dir / "tables",
    )


if __name__ == "__main__":
    main()
