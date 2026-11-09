"""Study random-start variability and compare it with PyTorch k-means."""

import csv
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gmm import GMMConfig, average_log_likelihood, fit_gmm
from src.utils.logging_utils import write_json
from experiments.task2.prepare_split import prepare_background_observations


SELECTED_COVARIANCE = "diagonal"
SELECTED_COMPONENTS = 4
RANDOM_SEEDS = tuple(range(20, 40))
KMEANS_SEED = 2026
REG_COVAR = 1e-4
MAX_ITERATIONS = 40


def _write_distribution_svg(values: list[float], path: Path) -> None:
    """Write a dependency-free SVG histogram of repeated-run log likelihoods."""
    width, height = 800, 460
    left, top, plot_width, plot_height = 90, 45, 660, 320
    lower = min(values)
    upper = max(values)
    if lower == upper:
        upper = lower + 1.0
    bin_count = min(10, max(5, round(len(values) ** 0.5)))
    counts = [0] * bin_count
    for value in values:
        index = min(
            bin_count - 1,
            int((value - lower) / (upper - lower) * bin_count),
        )
        counts[index] += 1
    maximum = max(counts)
    bar_width = plot_width / bin_count
    bars = []
    for index, count in enumerate(counts):
        bar_height = (count / maximum) * plot_height if maximum else 0
        x = left + index * bar_width + 3
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 6:.2f}" '
            f'height="{bar_height:.2f}" fill="#4776a8"/>'
        )
        bars.append(
            f'<text x="{x + (bar_width - 6) / 2:.2f}" y="{y - 7:.2f}" '
            f'text-anchor="middle" font-size="12">{count}</text>'
        )
    median_value = statistics.median(values)
    median_x = left + (median_value - lower) / (upper - lower) * plot_width
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="400" y="25" text-anchor="middle" font-size="18" font-family="sans-serif">Final training log-likelihood across random initializations</text>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
            *bars,
            f'<line x1="{median_x:.2f}" y1="{top}" x2="{median_x:.2f}" y2="{top + plot_height}" stroke="#c43b3b" stroke-dasharray="6 4"/>',
            f'<text x="{median_x:.2f}" y="{top + 15}" text-anchor="middle" font-size="12" fill="#a22">median</text>',
            f'<text x="{left + plot_width / 2:.2f}" y="420" text-anchor="middle" font-size="14">Final training log-likelihood per frame</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" transform="rotate(-90 20 {top + plot_height / 2:.2f})" text-anchor="middle" font-size="14">Run count</text>',
            f'<text x="{left}" y="{top + plot_height + 22}" font-size="12">{lower:.4f}</text>',
            f'<text x="{left + plot_width}" y="{top + plot_height + 22}" text-anchor="end" font-size="12">{max(values):.4f}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg + "\n", encoding="utf-8")


def run_initialization_study(
    fitting_data: torch.Tensor,
    heldout_data: torch.Tensor,
    tables_dir: Path,
    plots_dir: Path,
) -> dict[str, object]:
    """Run 20 separate random fits and a fair random-vs-k-means comparison."""
    config = GMMConfig(
        n_components=SELECTED_COMPONENTS,
        covariance_type=SELECTED_COVARIANCE,
        reg_covar=REG_COVAR,
        max_iter=MAX_ITERATIONS,
    )
    repeated = []
    for seed in RANDOM_SEEDS:
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
        repeated.append(
            {
                "seed": seed,
                "train_log_likelihood": result.log_likelihood,
                "train_average_log_likelihood": result.log_likelihood
                / fitting_data.shape[0],
                "heldout_average_log_likelihood": heldout_average,
                "iterations": result.iterations,
                "converged": result.converged,
            }
        )
        print(
            f"random initialization seed={seed}: "
            f"train LL/frame={repeated[-1]['train_average_log_likelihood']:.6f}, "
            f"heldout avg LL={heldout_average:.6f}"
        )

    train_values = [float(row["train_log_likelihood"]) for row in repeated]
    best_index = max(range(len(train_values)), key=train_values.__getitem__)
    worst_index = min(range(len(train_values)), key=train_values.__getitem__)
    first_index = 0
    summary = {
        "covariance_type": SELECTED_COVARIANCE,
        "n_components": SELECTED_COMPONENTS,
        "seed_count": len(RANDOM_SEEDS),
        "seeds": list(RANDOM_SEEDS),
        "best_train_log_likelihood": train_values[best_index],
        "best_seed": RANDOM_SEEDS[best_index],
        "worst_train_log_likelihood": train_values[worst_index],
        "worst_seed": RANDOM_SEEDS[worst_index],
        "median_train_log_likelihood": statistics.median(train_values),
        "single_run_seed": RANDOM_SEEDS[first_index],
        "single_run_train_log_likelihood": train_values[first_index],
        "best_of_n_improvement_over_single_run": (
            train_values[best_index] - train_values[first_index]
        ),
    }

    random_result = fit_gmm(
        fitting_data,
        config,
        seed=KMEANS_SEED,
        initialization="random",
    )
    kmeans_result = fit_gmm(
        fitting_data,
        config,
        seed=KMEANS_SEED,
        initialization="kmeans",
    )
    comparison = []
    for method, result in (
        ("random", random_result),
        ("kmeans", kmeans_result),
    ):
        comparison.append(
            {
                "initialization": method,
                "seed": KMEANS_SEED,
                "train_log_likelihood": result.log_likelihood,
                "train_average_log_likelihood": result.log_likelihood
                / fitting_data.shape[0],
                "heldout_average_log_likelihood": average_log_likelihood(
                    heldout_data,
                    result.model,
                    config.chunk_size,
                ),
                "iterations": result.iterations,
                "converged": result.converged,
            }
        )
    comparison.sort(
        key=lambda row: float(row["heldout_average_log_likelihood"]),
        reverse=True,
    )
    summary["initialization_comparison_winner_by_heldout"] = comparison[0][
        "initialization"
    ]

    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / "initialization_runs.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(repeated[0]))
        writer.writeheader()
        writer.writerows(repeated)
    with (tables_dir / "initialization_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    _write_distribution_svg(
        [float(row["train_average_log_likelihood"]) for row in repeated],
        plots_dir / "random_initialization_distribution.svg",
    )
    write_json(
        tables_dir / "initialization_summary.json",
        {
            "configuration": {
                "covariance_type": SELECTED_COVARIANCE,
                "n_components": SELECTED_COMPONENTS,
                "reg_covar": REG_COVAR,
                "max_iterations": MAX_ITERATIONS,
                "fitting_frames": fitting_data.shape[0],
                "heldout_frames": heldout_data.shape[0],
            },
            "repeated_random_initializations": summary,
            "random_vs_kmeans": comparison,
        },
    )
    return {
        "repeated": repeated,
        "comparison": comparison,
        "summary": summary,
    }


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / "task2"
    fitting_data, heldout_data, _ = prepare_background_observations(
        PROJECT_ROOT,
        results_dir / "tables",
    )
    run_initialization_study(
        fitting_data,
        heldout_data,
        results_dir / "tables",
        results_dir / "plots",
    )


if __name__ == "__main__":
    main()
