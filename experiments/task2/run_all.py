"""Run every Task 2 GMM behaviour experiment on background features only."""

import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import TextIO

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.task2.covariance_study import run_covariance_study
from experiments.task2.degeneracy_study import run_degeneracy_study
from experiments.task2.initialization_study import run_initialization_study
from experiments.task2.prepare_split import prepare_background_observations


SPLIT_SEED = 2026
FIT_FRAME_SEED = 2027
MAX_FIT_FRAMES_PER_SPEAKER = 16
HELDOUT_FRACTION = 0.20


class _Tee:
    """Write experiment output to both the terminal and persistent log."""

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


def _write_summary(
    results_dir: Path,
    split_metadata: dict[str, object],
    fitting_data,
    heldout_data,
    covariance_rows: list[dict[str, object]],
    initialization_result: dict[str, object],
    degeneracy_result: dict[str, object],
) -> None:
    """Write a concise report from the persisted experiment results."""
    best_bic = min(covariance_rows, key=lambda row: float(row["bic"]))
    best_heldout = max(
        covariance_rows,
        key=lambda row: float(row["heldout_average_log_likelihood"]),
    )
    init_summary = initialization_result["summary"]
    comparison = initialization_result["comparison"]

    lines = [
        "# Task 2 — GMM Behaviour Study",
        "",
        "## Data and split",
        "",
        "- Only Task 1 background features were loaded; development and evaluation data were not used.",
        "- Split method: deterministic `torch.randperm` of sorted unique speaker IDs, seed 2026.",
        f"- Speakers: {split_metadata['fitting_speaker_count']} fitting / {split_metadata['heldout_speaker_count']} held out "
        f"of {split_metadata['speaker_count']} total.",
        f"- Recordings: {split_metadata['fitting_recording_count']} fitting / {split_metadata['heldout_recording_count']} held out.",
        f"- Fitting observations: {fitting_data.shape[0]} frames, at most {MAX_FIT_FRAMES_PER_SPEAKER} sampled frames per fitting speaker "
        f"(seed {FIT_FRAME_SEED}); the held-out score uses all {heldout_data.shape[0]} held-out frames.",
        "- The split is speaker-disjoint and is saved in `tables/speaker_split.csv` and `tables/speaker_split.json`.",
        "",
        "## Covariance and component-count comparison",
        "",
        "AIC/BIC use the fitting log likelihood and the same sampled fitting observations for every model. Lower is preferred. "
        "Held-out average log likelihood is scored on all held-out background frames; higher is preferred.",
        "",
        "| Covariance | Components | Parameters | AIC | BIC | Held-out avg. log-likelihood | Iterations | Converged |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in covariance_rows:
        lines.append(
            f"| {row['covariance_type']} | {row['n_components']} | {row['n_parameters']} "
            f"| {float(row['aic']):.3f} | {float(row['bic']):.3f} "
            f"| {float(row['heldout_average_log_likelihood']):.6f} "
            f"| {row['iterations']} | {'yes' if row['converged'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"- Lowest BIC: **{best_bic['covariance_type']} / {best_bic['n_components']} components** "
            f"(BIC {float(best_bic['bic']):.3f}).",
            f"- Highest held-out average log likelihood: **{best_heldout['covariance_type']} / "
            f"{best_heldout['n_components']} components** "
            f"({float(best_heldout['heldout_average_log_likelihood']):.6f}).",
            "- These criteria can select different models; the BIC choice is a candidate for Task 3, not an instruction to ignore held-out behaviour.",
            "",
            "## Initialization behaviour",
            "",
            f"The repeated-start study used {init_summary['seed_count']} separate random fits of a diagonal, "
            f"{init_summary['n_components']}-component GMM. The plotted quantity is final training log likelihood per frame.",
            "",
            f"- Best: {float(init_summary['best_train_log_likelihood']):.6f} (total; seed {init_summary['best_seed']}).",
            f"- Median: {float(init_summary['median_train_log_likelihood']):.6f} (total).",
            f"- Worst: {float(init_summary['worst_train_log_likelihood']):.6f} (total; seed {init_summary['worst_seed']}).",
            f"- First single run (seed {init_summary['single_run_seed']}): "
            f"{float(init_summary['single_run_train_log_likelihood']):.6f} (total).",
            f"- Best-of-20 improvement over that single run: "
            f"{float(init_summary['best_of_n_improvement_over_single_run']):.6f} total log-likelihood.",
            "- Selecting the best start helps because EM can converge to different local optima from different initial parameters.",
            "",
            "The SVG distribution plot is `plots/random_initialization_distribution.svg`; run-level values are in "
            "`tables/initialization_runs.csv`.",
            "",
            "### Random vs. k-means initialization",
            "",
            "| Initialization | Train avg. log-likelihood | Held-out avg. log-likelihood | Iterations | Converged |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['initialization']} | {float(row['train_average_log_likelihood']):.6f} "
            f"| {float(row['heldout_average_log_likelihood']):.6f} "
            f"| {row['iterations']} | {'yes' if row['converged'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"By held-out average log likelihood, **{init_summary['initialization_comparison_winner_by_heldout']}** "
            "initialization was better in this run. Both methods used the same diagonal, 4-component model, fitting set, "
            "regularization, and seed.",
            "",
            "## Degenerate component and remedy",
            "",
            "Eight identical 3-dimensional observations produce a zero singular covariance. Direct Cholesky factorization "
            f"reported info={degeneracy_result['unregularized_cholesky_info']}. Adding "
            f"`epsilon I` with epsilon={degeneracy_result['regularization_epsilon']:.1e} made Cholesky succeed "
            f"(info={degeneracy_result['regularized_cholesky_info']}); the resulting average log likelihood "
            f"was finite ({float(degeneracy_result['regularized_average_log_likelihood']):.6f}).",
            "",
            "## Carry-forward observation",
            "",
            f"The BIC-preferred Task 2 candidate is {best_bic['covariance_type']} covariance with "
            f"{best_bic['n_components']} components. Consider it alongside the held-out ranking before choosing a UBM in Task 3. "
            "All fitting and held-out comparisons used the Task 1 402-dimensional feature representation without alteration.",
            "",
            "## Reproducibility and files",
            "",
            "- Covariance grid: `tables/covariance_comparison.csv` and `.json`.",
            "- Initialization runs/comparison/summary: `tables/initialization_runs.csv`, "
            "`tables/initialization_comparison.csv`, and `tables/initialization_summary.json`.",
            "- Degeneracy details: `tables/degeneracy_demo.json`.",
            "- Execution output: `logs/run_all.log`.",
            "- GMM fitting uses PyTorch only; the distribution plot is generated directly as SVG so no plotting package is required.",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all() -> None:
    results_dir = PROJECT_ROOT / "results" / "task2"
    tables_dir = results_dir / "tables"
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "run_all.log").open("w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        with redirect_stdout(_Tee(original_stdout, log_file)):
            fitting_data, heldout_data, split_metadata = prepare_background_observations(
                PROJECT_ROOT,
                tables_dir,
                heldout_fraction=HELDOUT_FRACTION,
                split_seed=SPLIT_SEED,
                max_fit_frames_per_speaker=MAX_FIT_FRAMES_PER_SPEAKER,
                fit_frame_seed=FIT_FRAME_SEED,
            )
            if not torch.isfinite(fitting_data).all() or not torch.isfinite(heldout_data).all():
                raise FloatingPointError("Prepared GMM data contains non-finite values")

            print(
                f"Loaded {split_metadata['recording_count']} background recordings; "
                f"fitting={split_metadata['fitting_speaker_count']} speakers/"
                f"{fitting_data.shape[0]} frames; "
                f"heldout={split_metadata['heldout_speaker_count']} speakers/"
                f"{heldout_data.shape[0]} frames."
            )
            covariance_rows = run_covariance_study(
                fitting_data,
                heldout_data,
                tables_dir,
            )
            initialization_result = run_initialization_study(
                fitting_data,
                heldout_data,
                tables_dir,
                results_dir / "plots",
            )
            degeneracy_result = run_degeneracy_study(
                tables_dir / "degeneracy_demo.json"
            )
            _write_summary(
                results_dir,
                split_metadata,
                fitting_data,
                heldout_data,
                covariance_rows,
                initialization_result,
                degeneracy_result,
            )
            print(f"Task 2 experiments completed. Report: {results_dir / 'summary.md'}")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
