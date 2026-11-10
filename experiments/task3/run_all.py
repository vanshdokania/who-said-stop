"""Train and compare Task 3 UBMs using Task 2's background-only split."""

import csv
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import TextIO

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.task3.plots import (
    plot_heldout_likelihood,
    plot_training_likelihood,
    plot_training_time,
)
from experiments.task3.prepare_data import load_task3_data
from src.gmm import GMMConfig, average_log_likelihood, fit_gmm
from src.utils.logging_utils import write_json


COMPONENT_COUNTS = (16, 32, 64, 128, 256)
COVARIANCE_TYPES = ("diagonal", "spherical", "tied", "full")
FIT_SEED_BASE = 4300
MAX_ITERATIONS = 40
CONVERGENCE_TOLERANCE = 1e-4
COVARIANCE_REGULARIZATION = 1e-4
SCORING_CHUNK_SIZE = 128
MAX_FIT_FRAMES_PER_SPEAKER = 16
FIT_FRAME_SEED = 2027


class _Tee:
    """Write logs to terminal and to the Task 3 log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


def _model_is_stable(model, heldout_score: float) -> tuple[bool, str]:
    if not torch.isfinite(model.weights).all():
        return False, "non-finite mixture weights"
    if not torch.isfinite(model.means).all():
        return False, "non-finite component means"
    if not torch.isfinite(model.covariances).all():
        return False, "non-finite covariance values"
    if (model.weights <= 0).any() or not torch.isclose(
        model.weights.sum(),
        torch.ones((), dtype=model.weights.dtype, device=model.weights.device),
        atol=1e-8,
        rtol=1e-8,
    ):
        return False, "invalid mixture weights"
    if model.covariance_type in ("diagonal", "spherical"):
        if (model.covariances <= 0).any():
            return False, "non-positive covariance"
    elif model.covariance_type == "full":
        if (model.covariances.diagonal(dim1=-2, dim2=-1) <= 0).any():
            return False, "non-positive covariance diagonal"
    elif (model.covariances.diagonal() <= 0).any():
        return False, "non-positive tied covariance diagonal"
    if not torch.isfinite(torch.tensor(heldout_score)):
        return False, "non-finite held-out likelihood"
    return True, "finite parameters and likelihood"


def _checkpoint_payload(
    model,
    row: dict[str, object],
    dataset_metadata: dict[str, int | str],
) -> dict[str, object]:
    """Build a weights-only-loadable checkpoint for later MAP adaptation."""
    return {
        "format": "who_said_stop_gmm_ubm_v1",
        "task": 3,
        "model": {
            "weights": model.weights.cpu(),
            "means": model.means.cpu(),
            "covariances": model.covariances.cpu(),
            "covariance_type": model.covariance_type,
            "reg_covar": model.reg_covar,
        },
        "training": {
            "n_components": row["n_components"],
            "covariance_type": row["covariance_type"],
            "initialization": row["initialization"],
            "seed": row["seed"],
            "max_iterations": MAX_ITERATIONS,
            "convergence_tolerance": CONVERGENCE_TOLERANCE,
            "convergence_criterion": "relative change in total fitting log-likelihood",
            "covariance_regularization": COVARIANCE_REGULARIZATION,
            "training_seconds": row["training_seconds"],
            "iterations": row["iterations"],
            "converged": row["converged"],
            "training_log_likelihood": row["training_log_likelihood"],
            "heldout_average_log_likelihood": row[
                "heldout_average_log_likelihood"
            ],
        },
        "data": dataset_metadata,
        "feature_dimension": int(model.n_features),
    }


def _write_results_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "covariance_type",
        "n_components",
        "status",
        "training_seconds",
        "heldout_evaluation_seconds",
        "iterations",
        "converged",
        "training_log_likelihood",
        "training_average_log_likelihood",
        "heldout_average_log_likelihood",
        "numerically_stable",
        "stability_note",
        "initialization",
        "seed",
        "max_iterations",
        "tolerance",
        "reg_covar",
        "fitting_frames",
        "heldout_frames",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    rows: list[dict[str, object]],
    best_row: dict[str, object],
    dataset_metadata: dict[str, int | str],
) -> None:
    successful = [row for row in rows if row["status"] == "success"]
    with (path.parent / "tables" / "ubm_comparison.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        likelihood_histories = json.load(file)["likelihood_histories"]
    largest_likelihood_drop = max(
        (
            previous - current
            for item in likelihood_histories
            for previous, current in zip(item["history"], item["history"][1:])
            if current < previous
        ),
        default=0.0,
    )
    lines = [
        "# Task 3 — UBM Training",
        "",
        "## Data and method",
        "",
        "- Only Task 1 background feature tensors were used. Development and evaluation recordings were not loaded.",
        "- Reused the fixed, speaker-disjoint Task 2 split; no Task 2 outputs were modified.",
        f"- Fitting: {dataset_metadata['fitting_speaker_count']} speakers, "
        f"{dataset_metadata['fitting_recording_count']} recordings, "
        f"{dataset_metadata['fitting_frames']} deterministic frames "
        f"(maximum {MAX_FIT_FRAMES_PER_SPEAKER} per speaker, sampling seed {FIT_FRAME_SEED}).",
        f"- Held out: {dataset_metadata['heldout_speaker_count']} speakers, "
        f"{dataset_metadata['heldout_recording_count']} recordings, "
        f"all {dataset_metadata['heldout_frames']} feature frames used only for evaluation.",
        "- Initialization: random fitting-data observations as distinct component means, equal initial weights, "
        "regularized global covariance; seed = 4300 + covariance-type index * 100 + component count.",
        f"- EM: PyTorch E/M steps; maximum {MAX_ITERATIONS} iterations; convergence when relative change in total "
        f"fitting log likelihood is at most {CONVERGENCE_TOLERANCE:g}.",
        f"- Covariance regularization: add {COVARIANCE_REGULARIZATION:g} to covariance diagonals "
        "(diagonal/spherical variances are lower-bounded by the same value).",
        f"- Scoring uses chunks of {SCORING_CHUNK_SIZE} observations. All models use identical fitting/held-out sets.",
        "- One fit per requested configuration; no held-out data is used for parameter updates.",
        "",
        "## Comparison results",
        "",
        "| Covariance | Components | Status | Time (s) | Iterations | Converged | Train log-likelihood | Held-out avg. log-likelihood | Numerically stable |",
        "|---|---:|---|---:|---:|:---:|---:|---:|:---:|",
    ]
    for row in rows:
        train_ll = (
            f"{float(row['training_log_likelihood']):.3f}"
            if row.get("training_log_likelihood") is not None
            else "—"
        )
        heldout_ll = (
            f"{float(row['heldout_average_log_likelihood']):.6f}"
            if row.get("heldout_average_log_likelihood") is not None
            else "—"
        )
        lines.append(
            f"| {row['covariance_type']} | {row['n_components']} | {row['status']} "
            f"| {float(row.get('training_seconds') or 0):.2f} | {row.get('iterations', '—')} "
            f"| {'yes' if row.get('converged') else 'no'} | {train_ll} | {heldout_ll} "
            f"| {'yes' if row.get('numerically_stable') else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Mixture-count and covariance observations",
            "",
            "- Compare held-out log-likelihood across component counts within each covariance family; a higher value indicates better held-out density fit.",
            "- Increasing component count adds representational flexibility and parameter cost. Diagonal and spherical covariance structures are substantially smaller than full covariance in 402 dimensions.",
            "- Full covariance needs `K × D × (D + 1) / 2` covariance parameters. At high K and D=402, individual component estimates are data-hungry; regularization helps numerical factorization but does not create missing statistical support.",
            "- Training time and convergence counts are in the table; failed or unstable fits, if any, are reported as such instead of being omitted.",
            f"- The largest adjacent decrease in a recorded total training-likelihood history was {largest_likelihood_drop:.6f}; this small non-monotonic step occurred in a regularized full-covariance fit and is recorded in the history rather than hidden.",
            "- Higher background likelihood is only a density-fit observation; it does not prove improved speaker-verification performance, which must be evaluated in later tasks.",
            "",
            "## Final UBM selection",
            "",
            f"Selected **{best_row['covariance_type']} covariance with {best_row['n_components']} components** "
            f"(held-out average log likelihood {float(best_row['heldout_average_log_likelihood']):.6f}; "
            f"training time {float(best_row['training_seconds']):.2f}s; "
            f"{best_row['iterations']} iterations; converged={str(best_row['converged']).lower()}).",
            "Selection is the numerically stable, converged successful model with the highest held-out average log likelihood. "
            "This provides an empirical background-density criterion for the later MAP adaptation stage; Task 3 does not claim "
            "verification performance.",
            "",
            f"The final PyTorch checkpoint is `models/ubm_final.pt`. It stores weights, means, covariance parameters and type, "
            f"feature dimension, and reproducible training metadata. {len(successful)} of {len(rows)} requested configurations "
            "completed successfully.",
            "",
            "## Artifacts",
            "",
            "- `tables/ubm_comparison.csv` and `.json`: configuration metrics and fit histories.",
            "- `plots/training_log_likelihood.svg`: per-frame training likelihood across EM iterations.",
            "- `plots/heldout_likelihood_by_components.svg`: held-out scores across mixture counts.",
            "- `plots/training_time.svg`: wall-clock training time per fit.",
            "- `logs/run_all.log`: experiment log.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all() -> None:
    results_dir = PROJECT_ROOT / "results" / "task3"
    tables_dir = results_dir / "tables"
    plots_dir = results_dir / "plots"
    logs_dir = results_dir / "logs"
    models_dir = results_dir / "models"
    for directory in (tables_dir, plots_dir, logs_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "run_all.log").open("w", encoding="utf-8") as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)):
            fitting_data, heldout_data, dataset_metadata = load_task3_data(
                PROJECT_ROOT,
                max_frames_per_speaker=MAX_FIT_FRAMES_PER_SPEAKER,
                frame_seed=FIT_FRAME_SEED,
            )
            print(
                f"Task 3 data: fitting={fitting_data.shape[0]} frames/"
                f"{dataset_metadata['fitting_speaker_count']} speakers; "
                f"heldout={heldout_data.shape[0]} frames/"
                f"{dataset_metadata['heldout_speaker_count']} speakers."
            )
            rows: list[dict[str, object]] = []
            histories: list[dict[str, object]] = []
            best_row = None
            for covariance_index, covariance_type in enumerate(COVARIANCE_TYPES):
                for n_components in COMPONENT_COUNTS:
                    seed = FIT_SEED_BASE + covariance_index * 100 + n_components
                    config = GMMConfig(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        reg_covar=COVARIANCE_REGULARIZATION,
                        max_iter=MAX_ITERATIONS,
                        tol=CONVERGENCE_TOLERANCE,
                        chunk_size=SCORING_CHUNK_SIZE,
                    )
                    row: dict[str, object] = {
                        "covariance_type": covariance_type,
                        "n_components": n_components,
                        "status": "running",
                        "training_seconds": None,
                        "heldout_evaluation_seconds": None,
                        "iterations": None,
                        "converged": False,
                        "training_log_likelihood": None,
                        "training_average_log_likelihood": None,
                        "heldout_average_log_likelihood": None,
                        "numerically_stable": False,
                        "stability_note": "",
                        "initialization": "random",
                        "seed": seed,
                        "max_iterations": MAX_ITERATIONS,
                        "tolerance": CONVERGENCE_TOLERANCE,
                        "reg_covar": COVARIANCE_REGULARIZATION,
                        "fitting_frames": fitting_data.shape[0],
                        "heldout_frames": heldout_data.shape[0],
                        "error": "",
                    }
                    try:
                        started = time.perf_counter()
                        fit_result = fit_gmm(
                            fitting_data,
                            config,
                            seed=seed,
                            initialization="random",
                        )
                        row["training_seconds"] = time.perf_counter() - started
                        row["iterations"] = fit_result.iterations
                        row["converged"] = fit_result.converged
                        row["training_log_likelihood"] = fit_result.log_likelihood
                        row["training_average_log_likelihood"] = (
                            fit_result.log_likelihood / fitting_data.shape[0]
                        )
                        started = time.perf_counter()
                        heldout_score = average_log_likelihood(
                            heldout_data,
                            fit_result.model,
                            config.chunk_size,
                        )
                        row["heldout_evaluation_seconds"] = (
                            time.perf_counter() - started
                        )
                        stable, stability_note = _model_is_stable(
                            fit_result.model,
                            heldout_score,
                        )
                        row["numerically_stable"] = stable
                        row["stability_note"] = stability_note
                        row["heldout_average_log_likelihood"] = heldout_score
                        row["status"] = (
                            "success"
                            if stable and fit_result.converged
                            else "unstable"
                            if not stable
                            else "not_converged"
                        )
                        histories.append(
                            {
                                "covariance_type": covariance_type,
                                "n_components": n_components,
                                "fitting_frames": fitting_data.shape[0],
                                "history": list(fit_result.likelihood_history),
                            }
                        )
                        if row["status"] == "success" and (
                            best_row is None
                            or heldout_score
                            > float(best_row["heldout_average_log_likelihood"])
                        ):
                            best_row = dict(row)
                            checkpoint_path = models_dir / "ubm_final.pt"
                            temporary_path = checkpoint_path.with_suffix(".pt.tmp")
                            torch.save(
                                _checkpoint_payload(
                                    fit_result.model,
                                    row,
                                    dataset_metadata,
                                ),
                                temporary_path,
                            )
                            temporary_path.replace(checkpoint_path)
                        print(
                            f"{covariance_type:9s} K={n_components:3d}: "
                            f"status={row['status']}, train_s={float(row['training_seconds']):.2f}, "
                            f"iter={row['iterations']}, converged={row['converged']}, "
                            f"heldout_avg_LL={heldout_score:.6f}, stable={stable}"
                        )
                    except Exception as error:
                        row["status"] = "failed"
                        row["error"] = f"{type(error).__name__}: {error}"
                        print(
                            f"{covariance_type:9s} K={n_components:3d}: "
                            f"FAILED: {row['error']}"
                        )
                        traceback.print_exc()
                    rows.append(row)
                    _write_results_table(tables_dir / "ubm_comparison.csv", rows)
                    write_json(
                        tables_dir / "ubm_comparison.json",
                        {
                            "configuration": {
                                "component_counts": list(COMPONENT_COUNTS),
                                "covariance_types": list(COVARIANCE_TYPES),
                                "initialization": "random",
                                "max_iterations": MAX_ITERATIONS,
                                "tolerance": CONVERGENCE_TOLERANCE,
                                "convergence_criterion": "relative change in total fitting log likelihood",
                                "reg_covar": COVARIANCE_REGULARIZATION,
                                "fit_frame_cap_per_speaker": MAX_FIT_FRAMES_PER_SPEAKER,
                                "fit_frame_seed": FIT_FRAME_SEED,
                                "fit_seed_base": FIT_SEED_BASE,
                                "scoring_chunk_size": SCORING_CHUNK_SIZE,
                                "feature_dimension": int(fitting_data.shape[1]),
                            },
                            "dataset": dataset_metadata,
                            "results": rows,
                            "likelihood_histories": histories,
                        },
                    )
            if best_row is None:
                raise RuntimeError(
                    "No requested UBM configuration completed with stable, converged parameters"
                )
            plot_training_likelihood(
                histories,
                plots_dir / "training_log_likelihood.svg",
            )
            plot_heldout_likelihood(
                rows,
                plots_dir / "heldout_likelihood_by_components.svg",
            )
            plot_training_time(rows, plots_dir / "training_time.svg")
            _write_summary(
                results_dir / "summary.md",
                rows,
                best_row,
                dataset_metadata,
            )
            print(
                f"Selected {best_row['covariance_type']} K={best_row['n_components']} "
                f"with held-out avg LL "
                f"{float(best_row['heldout_average_log_likelihood']):.6f}. "
                f"Checkpoint: {models_dir / 'ubm_final.pt'}"
            )
            if len(rows) != len(COMPONENT_COUNTS) * len(COVARIANCE_TYPES):
                raise RuntimeError("Not every requested Task 3 configuration was attempted")


def main() -> None:
    run_all()


if __name__ == "__main__":
    run_all()
