"""Run Task 4 MAP adaptation, development comparisons, and eval enrolment."""

import csv
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.task4.data import (
    EnrolmentSpeaker,
    load_development_trials,
    load_enrolment_speakers,
)
from experiments.task4.evaluation import (
    score_claimed_speaker_trials,
    verification_metrics,
)
from experiments.task4.plots import plot_relevance_metrics
from src.gmm import (
    GMMConfig,
    GaussianMixture,
    adapt_means_map,
    fit_gmm,
)
from src.utils.logging_utils import write_json


UBM_PATH = PROJECT_ROOT / "results" / "task3" / "models" / "ubm_final.pt"
RELEVANCE_FACTORS = (4, 8, 16, 32)
INDEPENDENT_COMPONENTS = 64
MAX_INDEPENDENT_ITERATIONS = 40
INDEPENDENT_TOLERANCE = 1e-4
REG_COVAR = 1e-4
OCCUPANCY_EPSILON = 1e-8
STATISTICS_CHUNK_SIZE = 256
SCORING_CHUNK_SIZE = 128
INDEPENDENT_SEED_BASE = 6400


def load_ubm(path: Path) -> tuple[GaussianMixture, dict[str, object]]:
    """Load and validate the Task 3 diagonal 64-component UBM checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Task 3 UBM checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "who_said_stop_gmm_ubm_v1":
        raise ValueError(f"Unsupported UBM checkpoint format: {path}")
    parameters = checkpoint.get("model")
    if not isinstance(parameters, dict):
        raise ValueError("UBM checkpoint has no model parameter dictionary")
    model = GaussianMixture(
        weights=parameters["weights"].to(torch.float64),
        means=parameters["means"].to(torch.float64),
        covariances=parameters["covariances"].to(torch.float64),
        covariance_type=parameters["covariance_type"],
        reg_covar=float(parameters["reg_covar"]),
    )
    if (
        checkpoint.get("feature_dimension") != 402
        or model.n_components != 64
        or model.n_features != 402
        or model.covariance_type != "diagonal"
    ):
        raise ValueError(
            "Task 4 requires the Task 3 64-component diagonal UBM "
            "with 402-dimensional features"
        )
    if model.covariances.shape != model.means.shape:
        raise ValueError("Diagonal UBM covariance tensor must have shape [K, D]")
    if (
        not torch.isfinite(model.weights).all()
        or not torch.isfinite(model.means).all()
        or not torch.isfinite(model.covariances).all()
        or (model.weights <= 0).any()
        or (model.covariances <= 0).any()
        or not torch.isclose(
            model.weights.sum(),
            torch.ones((), dtype=model.weights.dtype),
            atol=1e-8,
            rtol=1e-8,
        )
    ):
        raise ValueError("Task 3 UBM contains invalid parameters")
    return model, checkpoint


def _validate_enrolment_speakers(
    speakers: dict[str, EnrolmentSpeaker],
) -> None:
    for speaker_id, speaker in speakers.items():
        if speaker.features.ndim != 2 or speaker.features.shape[1] != 402:
            raise ValueError(f"Invalid feature shape for {speaker_id}")
        if speaker.features.shape[0] == 0 or not torch.isfinite(speaker.features).all():
            raise ValueError(f"Empty or non-finite enrolment data for {speaker_id}")


def _save_map_models(
    path: Path,
    models: dict[str, dict[str, object]],
    ubm: GaussianMixture,
    split: str,
    relevance_factor: int,
) -> None:
    """Save speaker means and shared unchanged UBM weights/covariances."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "who_said_stop_map_mean_models_v1",
        "split": split,
        "adaptation": "mean_only_map",
        "relevance_factor": relevance_factor,
        "occupancy_epsilon": OCCUPANCY_EPSILON,
        "covariance_type": ubm.covariance_type,
        "reg_covar": ubm.reg_covar,
        "feature_dimension": ubm.n_features,
        "n_components": ubm.n_components,
        "weights": ubm.weights.clone(),
        "covariances": ubm.covariances.clone(),
        "speakers": models,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _map_speaker_record(
    speaker: EnrolmentSpeaker,
    ubm: GaussianMixture,
    relevance_factor: int,
) -> tuple[dict[str, object], dict[str, object]]:
    result = adapt_means_map(
        speaker.features,
        ubm,
        relevance_factor,
        occupancy_epsilon=OCCUPANCY_EPSILON,
        chunk_size=STATISTICS_CHUNK_SIZE,
    )
    if not torch.equal(result.model.weights, ubm.weights):
        raise RuntimeError("Mean-only MAP adaptation changed UBM weights")
    if not torch.equal(result.model.covariances, ubm.covariances):
        raise RuntimeError("Mean-only MAP adaptation changed UBM covariances")
    model_record = {
        "means": result.model.means,
        "enrolment_file_ids": list(speaker.file_ids),
        "enrolment_indices": list(speaker.enrolment_indices),
        "pooled_frame_count": int(speaker.features.shape[0]),
        "occupancies": result.statistics.occupancies,
        "zero_occupancy_components": result.retained_components,
    }
    occupancy = result.statistics.occupancies
    stats_record = {
        "split": "",
        "relevance_factor": relevance_factor,
        "speaker_id": speaker.speaker_id,
        "components": ubm.n_components,
        "enrolment_recordings": len(speaker.file_ids),
        "pooled_frames": int(speaker.features.shape[0]),
        "zero_occupancy_components": result.retained_components,
        "min_occupancy": float(occupancy.min().item()),
        "max_occupancy": float(occupancy.max().item()),
        "means_changed_components": int(
            (result.model.means != ubm.means).any(dim=1).sum().item()
        ),
    }
    return model_record, stats_record


def _train_independent_models(
    speakers: dict[str, EnrolmentSpeaker],
) -> tuple[dict[str, GaussianMixture], list[dict[str, object]]]:
    """Fit one independent 64-component diagonal GMM per dev speaker."""
    models = {}
    rows = []
    config = GMMConfig(
        n_components=INDEPENDENT_COMPONENTS,
        covariance_type="diagonal",
        reg_covar=REG_COVAR,
        max_iter=MAX_INDEPENDENT_ITERATIONS,
        tol=INDEPENDENT_TOLERANCE,
        chunk_size=STATISTICS_CHUNK_SIZE,
    )
    for index, speaker_id in enumerate(sorted(speakers)):
        speaker = speakers[speaker_id]
        seed = INDEPENDENT_SEED_BASE + index
        started = time.perf_counter()
        fit = fit_gmm(
            speaker.features,
            config,
            seed=seed,
            initialization="random",
        )
        elapsed = time.perf_counter() - started
        if (
            not torch.isfinite(fit.model.means).all()
            or not torch.isfinite(fit.model.weights).all()
            or not torch.isfinite(fit.model.covariances).all()
        ):
            raise FloatingPointError(
                f"Independent GMM for {speaker_id} has non-finite parameters"
            )
        models[speaker_id] = fit.model
        rows.append(
            {
                "speaker_id": speaker_id,
                "pooled_frames": int(speaker.features.shape[0]),
                "n_components": INDEPENDENT_COMPONENTS,
                "iterations": fit.iterations,
                "converged": fit.converged,
                "training_seconds": elapsed,
                "training_log_likelihood": fit.log_likelihood,
                "seed": seed,
            }
        )
        if (index + 1) % 20 == 0 or index + 1 == len(speakers):
            print(f"Independent speaker GMMs: {index + 1}/{len(speakers)}")
    return models, rows


def _save_independent_models(
    path: Path,
    models: dict[str, GaussianMixture],
    fit_rows: list[dict[str, object]],
) -> None:
    payload = {
        "format": "who_said_stop_independent_speaker_gmms_v1",
        "split": "dev",
        "covariance_type": "diagonal",
        "n_components": INDEPENDENT_COMPONENTS,
        "reg_covar": REG_COVAR,
        "feature_dimension": 402,
        "speakers": {
            speaker_id: {
                "weights": model.weights,
                "means": model.means,
                "covariances": model.covariances,
                "covariance_type": model.covariance_type,
                "reg_covar": model.reg_covar,
            }
            for speaker_id, model in models.items()
        },
        "fit_metadata": fit_rows,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _write_trial_scores(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("method", "relevance_factor", "trial_id", "claimed_id", "test_file", "label", "score"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    relevance_rows: list[dict[str, object]],
    independent_metrics: dict[str, float | int],
    data_summary: dict[str, object],
    statistics_rows: list[dict[str, object]],
) -> None:
    selected = min(relevance_rows, key=lambda row: float(row["eer"]))
    zero_components = sum(
        int(row["zero_occupancy_components"])
        for row in statistics_rows
        if row["split"] == "dev"
    )
    total_dev_components = sum(
        int(row["components"])
        for row in statistics_rows
        if row["split"] == "dev"
    )
    lines = [
        "# Task 4 — MAP Adaptation and Enrolment",
        "",
        "## Data and method",
        "",
        "- Loaded the existing 64-component, 402-dimensional diagonal UBM from `results/task3/models/ubm_final.pt`; no UBM retraining was performed.",
        f"- Development: {data_summary['dev_speakers']} speakers, {data_summary['dev_recordings']} enrolment recordings, "
        f"{data_summary['dev_frames']} pooled frames.",
        f"- Evaluation enrolment: {data_summary['eval_speakers']} speakers, {data_summary['eval_recordings']} recordings, "
        f"{data_summary['eval_frames']} pooled frames. No evaluation trial data or labels were read.",
        "- For each speaker, recordings were ordered by `enrol_index`, loaded with the unchanged Task 1 feature representation, and concatenated into one frame sequence.",
        "- UBM responsibilities were computed in the log domain. Zero-order occupancies are sums of responsibilities; first-order statistics are responsibility-weighted feature sums.",
        f"- Mean-only MAP adaptation uses `mu_hat = (F + r * mu) / (N + r)` with negligible occupancy threshold {OCCUPANCY_EPSILON:g}. "
        "Components at or below the threshold retain their UBM means. UBM weights and diagonal covariances are copied unchanged.",
        "",
        "## Relevance-factor limits",
        "",
        "As r approaches zero, a component with nonzero occupancy approaches its enrolment-data mean `F/N`; an empty/negligible component retains the UBM mean. "
        "As r approaches infinity, every adapted mean approaches its UBM mean. Thus r controls the balance: smaller values let limited enrolment evidence move means more, while larger values shrink them more strongly toward background knowledge.",
        "",
        "## Development trial comparison",
        "",
        "Development models were scored only on `dev_trials.csv`. Each score is the claimed-speaker average log likelihood minus the fixed UBM average log likelihood, using the test utterance features. "
        "EER is estimated by sweeping unique score thresholds; FAR and FRR are reported at the closest operating point. Evaluation trials and labels were not used.",
        "",
        "| Model | Relevance factor | EER | FAR at EER | FRR at EER | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in relevance_rows:
        lines.append(
            f"| MAP mean-only | {row['relevance_factor']} | {float(row['eer']):.6f} "
            f"| {float(row['far_at_eer']):.6f} | {float(row['frr_at_eer']):.6f} "
            f"| {float(row['threshold_at_eer']):.6f} |"
        )
    lines.append(
        f"| Independent diagonal GMM ({INDEPENDENT_COMPONENTS} components) | — "
        f"| {float(independent_metrics['eer']):.6f} | {float(independent_metrics['far_at_eer']):.6f} "
        f"| {float(independent_metrics['frr_at_eer']):.6f} "
        f"| {float(independent_metrics['threshold_at_eer']):.6f} |"
    )
    lines.extend(
        [
            "",
            f"Selected relevance factor: **r={selected['relevance_factor']}**, the lowest development EER "
            f"({float(selected['eer']):.6f}) among the four required values.",
            "The best tested value is at the upper end of the requested grid; no relevance factors above 32 were tested, so selection is limited to the Task 4 candidates.",
            "",
            "### MAP versus independently fitted speaker GMMs",
            "",
            f"The independent baseline fits a separate diagonal {INDEPENDENT_COMPONENTS}-component GMM from scratch for each development speaker, using the same pooled enrolment frames, feature representation, and diagonal covariance regularization. "
            f"Its development EER was {float(independent_metrics['eer']):.6f}. MAP starts from the shared background UBM and only moves means, which constrains estimates for speakers with limited enrolment data; an independent model must estimate all weights, means, and variances from those limited frames.",
            f"The independent 64-component result is a controlled same-size baseline, but each speaker has only three enrolment recordings, with {data_summary['dev_min_speaker_frames']}–{data_summary['dev_max_speaker_frames']} pooled frames (median {data_summary['dev_median_speaker_frames']:.0f}). It has much less data per component than the UBM. Development EER—not training likelihood—determines the relevance factor.",
            "Per-speaker baseline convergence and fit times are listed in `tables/independent_gmm_fits.csv`.",
            "",
            "## Zero-occupancy handling and invariants",
            "",
            f"Across the development adaptations, {zero_components} of {total_dev_components} speaker/component occupancies were zero or below the configured threshold. Such components retained their exact UBM mean. "
            "Every saved MAP model stores the shared original UBM weights and covariances; runtime checks required exact tensor equality before saving.",
            "",
            "## Evaluation enrolment models",
            "",
            f"Applied the selected development relevance factor r={selected['relevance_factor']} to evaluation enrolment recordings only. "
            "These speaker models are saved for later use; they were not scored against evaluation trials.",
            "",
            "## Artifacts",
            "",
            "- Relevance-factor metrics: `tables/relevance_factor_comparison.csv`.",
            "- Development comparison including independent baseline: `tables/development_comparison.csv`.",
            "- Speaker/component occupancy details: `tables/speaker_statistics.csv`.",
            "- Development trial scores: `tables/development_trial_scores.csv`.",
            "- MAP development models: `models/development_map_r4.pt`, `development_map_r8.pt`, `development_map_r16.pt`, and `development_map_r32.pt`.",
            "- Independent development GMMs: `models/development_independent_gmms.pt`.",
            f"- Evaluation MAP models: `models/evaluation_map_r{selected['relevance_factor']}.pt`.",
            "- Development EER plot: `plots/development_eer_by_relevance.svg`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all() -> None:
    results_dir = PROJECT_ROOT / "results" / "task4"
    tables_dir = results_dir / "tables"
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"
    for directory in (tables_dir, models_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ubm, _ = load_ubm(UBM_PATH)
    original_weights = ubm.weights.clone()
    original_covariances = ubm.covariances.clone()
    development_speakers = load_enrolment_speakers(PROJECT_ROOT, "dev")
    _validate_enrolment_speakers(development_speakers)
    trials, test_feature_cache = load_development_trials(PROJECT_ROOT)
    if {trial.claimed_id for trial in trials} - set(development_speakers):
        raise ValueError("A development trial claims a speaker without enrolment data")

    feature_rows = []
    statistics_rows = []
    relevance_rows = []
    score_rows = []
    for relevance_factor in RELEVANCE_FACTORS:
        started = time.perf_counter()
        speaker_models = {}
        factor_statistics = []
        for speaker_id in sorted(development_speakers):
            model_record, stats_record = _map_speaker_record(
                development_speakers[speaker_id],
                ubm,
                relevance_factor,
            )
            if not torch.isfinite(model_record["means"]).all():
                raise FloatingPointError(f"Non-finite MAP means for {speaker_id}")
            speaker_models[speaker_id] = model_record
            stats_record["split"] = "dev"
            factor_statistics.append(stats_record)
        adaptation_seconds = time.perf_counter() - started
        model_map = {
            speaker_id: GaussianMixture(
                weights=original_weights,
                means=record["means"],
                covariances=original_covariances,
                covariance_type=ubm.covariance_type,
                reg_covar=ubm.reg_covar,
            )
            for speaker_id, record in speaker_models.items()
        }
        scored = score_claimed_speaker_trials(
            trials,
            test_feature_cache,
            model_map,
            ubm,
            SCORING_CHUNK_SIZE,
        )
        metrics = verification_metrics(scored)
        relevance_rows.append(
            {
                "method": "MAP",
                "relevance_factor": relevance_factor,
                **metrics,
                "adaptation_seconds": adaptation_seconds,
                "speaker_count": len(speaker_models),
                "zero_occupancy_components": sum(
                    int(record["zero_occupancy_components"])
                    for record in speaker_models.values()
                ),
            }
        )
        for score in scored:
            score_rows.append(
                {
                    "method": "MAP",
                    "relevance_factor": relevance_factor,
                    **score,
                }
            )
        statistics_rows.extend(factor_statistics)
        _save_map_models(
            models_dir / f"development_map_r{relevance_factor}.pt",
            speaker_models,
            ubm,
            "dev",
            relevance_factor,
        )
        print(
            f"MAP r={relevance_factor}: EER={float(metrics['eer']):.6f}, "
            f"FAR={float(metrics['far_at_eer']):.6f}, "
            f"FRR={float(metrics['frr_at_eer']):.6f}, "
            f"adapt_s={adaptation_seconds:.2f}"
        )

    independent_models, independent_fit_rows = _train_independent_models(
        development_speakers
    )
    _save_independent_models(
        models_dir / "development_independent_gmms.pt",
        independent_models,
        independent_fit_rows,
    )
    independent_scores = score_claimed_speaker_trials(
        trials,
        test_feature_cache,
        independent_models,
        ubm,
        SCORING_CHUNK_SIZE,
    )
    independent_metrics = verification_metrics(independent_scores)
    for score in independent_scores:
        score_rows.append(
            {
                "method": "independent_gmm",
                "relevance_factor": "",
                **score,
            }
        )
    _write_trial_scores(
        tables_dir / "development_trial_scores.csv",
        score_rows,
    )

    relevance_fields = list(relevance_rows[0])
    with (tables_dir / "relevance_factor_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=relevance_fields)
        writer.writeheader()
        writer.writerows(relevance_rows)
    development_comparison = [
        {
            "method": "MAP",
            "relevance_factor": row["relevance_factor"],
            "speaker_count": row["speaker_count"],
            "eer": row["eer"],
            "far_at_eer": row["far_at_eer"],
            "frr_at_eer": row["frr_at_eer"],
            "threshold_at_eer": row["threshold_at_eer"],
            "total_trials": row["total_trials"],
            "target_trials": row["target_trials"],
            "nontarget_trials": row["nontarget_trials"],
        }
        for row in relevance_rows
    ]
    development_comparison.append(
        {
            "method": "independent_gmm",
            "relevance_factor": "",
            "speaker_count": len(independent_models),
            "eer": independent_metrics["eer"],
            "far_at_eer": independent_metrics["far_at_eer"],
            "frr_at_eer": independent_metrics["frr_at_eer"],
            "threshold_at_eer": independent_metrics["threshold_at_eer"],
            "total_trials": independent_metrics["total_trials"],
            "target_trials": independent_metrics["target_trials"],
            "nontarget_trials": independent_metrics["nontarget_trials"],
        }
    )
    with (tables_dir / "development_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(development_comparison[0]),
        )
        writer.writeheader()
        writer.writerows(development_comparison)

    speaker_statistics_fields = list(statistics_rows[0])
    with (tables_dir / "speaker_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=speaker_statistics_fields)
        writer.writeheader()
        writer.writerows(statistics_rows)
    with (tables_dir / "independent_gmm_fits.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(independent_fit_rows[0]))
        writer.writeheader()
        writer.writerows(independent_fit_rows)

    selected_row = min(relevance_rows, key=lambda row: float(row["eer"]))
    selected_relevance = int(selected_row["relevance_factor"])
    evaluation_speakers = load_enrolment_speakers(PROJECT_ROOT, "eval")
    _validate_enrolment_speakers(evaluation_speakers)
    eval_models = {}
    eval_stats = []
    for speaker_id in sorted(evaluation_speakers):
        model_record, stats_record = _map_speaker_record(
            evaluation_speakers[speaker_id],
            ubm,
            selected_relevance,
        )
        if not torch.isfinite(model_record["means"]).all():
            raise FloatingPointError(f"Non-finite evaluation MAP means for {speaker_id}")
        eval_models[speaker_id] = model_record
        stats_record["split"] = "eval"
        eval_stats.append(stats_record)
    _save_map_models(
        models_dir / f"evaluation_map_r{selected_relevance}.pt",
        eval_models,
        ubm,
        "eval",
        selected_relevance,
    )
    statistics_rows.extend(eval_stats)
    with (tables_dir / "speaker_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(statistics_rows[0]))
        writer.writeheader()
        writer.writerows(statistics_rows)

    data_summary: dict[str, object] = {
        "dev_speakers": len(development_speakers),
        "dev_recordings": sum(len(speaker.file_ids) for speaker in development_speakers.values()),
        "dev_frames": sum(speaker.features.shape[0] for speaker in development_speakers.values()),
        "eval_speakers": len(evaluation_speakers),
        "eval_recordings": sum(len(speaker.file_ids) for speaker in evaluation_speakers.values()),
        "eval_frames": sum(speaker.features.shape[0] for speaker in evaluation_speakers.values()),
        "dev_trials": len(trials),
        "dev_unique_test_files": len(test_feature_cache),
        "dev_min_speaker_frames": min(
            speaker.features.shape[0] for speaker in development_speakers.values()
        ),
        "dev_median_speaker_frames": float(
            torch.tensor(
                [
                    speaker.features.shape[0]
                    for speaker in development_speakers.values()
                ],
                dtype=torch.float64,
            ).median().item()
        ),
        "dev_max_speaker_frames": max(
            speaker.features.shape[0] for speaker in development_speakers.values()
        ),
    }
    _write_summary(
        results_dir / "summary.md",
        relevance_rows,
        independent_metrics,
        data_summary,
        statistics_rows,
    )
    plot_relevance_metrics(
        development_comparison,
        plots_dir / "development_eer_by_relevance.svg",
    )
    write_json(
        tables_dir / "task4_metadata.json",
        {
            "ubm_source": str(UBM_PATH.relative_to(PROJECT_ROOT)),
            "ubm_weights_unchanged": torch.equal(ubm.weights, original_weights),
            "ubm_covariances_unchanged": torch.equal(
                ubm.covariances,
                original_covariances,
            ),
            "relevance_factors": list(RELEVANCE_FACTORS),
            "selected_relevance_factor": selected_relevance,
            "independent_baseline": {
                "covariance_type": "diagonal",
                "n_components": INDEPENDENT_COMPONENTS,
                "reg_covar": REG_COVAR,
                "max_iterations": MAX_INDEPENDENT_ITERATIONS,
                "tolerance": INDEPENDENT_TOLERANCE,
                "converged_speaker_count": sum(
                    bool(row["converged"]) for row in independent_fit_rows
                ),
                "speaker_count": len(independent_fit_rows),
            },
            "map_settings": {
                "mode": "mean_only",
                "occupancy_epsilon": OCCUPANCY_EPSILON,
                "statistics_chunk_size": STATISTICS_CHUNK_SIZE,
                "scoring_chunk_size": SCORING_CHUNK_SIZE,
            },
            "data": data_summary,
            "development_comparison": development_comparison,
        },
    )
    print(
        f"Selected r={selected_relevance}; saved "
        f"{len(eval_models)} evaluation speaker models; "
        f"Task 4 outputs: {results_dir}"
    )


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
