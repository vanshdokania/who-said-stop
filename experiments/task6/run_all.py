"""Run the Task 6 development-only analysis and ablations."""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.task5.data import Trial, load_trials_and_features
from experiments.task5.scoring import (
    development_curve,
    load_ubm,
    score_trials,
)
from experiments.task6.plots import (
    plot_enrolment_size,
    plot_model_comparison,
    plot_relevance_factor,
)
from src.gmm import GaussianMixture, adapt_means_map


RESULTS_DIR = PROJECT_ROOT / "results" / "task6"
TASK4_COMPARISON = (
    PROJECT_ROOT / "results" / "task4" / "tables" / "development_comparison.csv"
)
UBM_PATH = PROJECT_ROOT / "results" / "task3" / "models" / "ubm_final.pt"
RELEVANCE_FACTORS = (4, 8, 16, 32)
ENROLMENT_SIZES = (1, 2, 3)
SELECTED_RELEVANCE_FACTOR = 32
OCCUPANCY_EPSILON = 1e-8
STATISTICS_CHUNK_SIZE = 256
SCORING_CHUNK_SIZE = 128


def _read_dev_enrolments() -> dict[str, list[tuple[int, str]]]:
    path = PROJECT_ROOT / "data" / "dev_enrol.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Development enrolment manifest is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"file_id", "speaker_id", "enrol_index"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
        seen_file_ids = set()
        for row_number, row in enumerate(reader, start=2):
            file_id = row["file_id"].strip()
            speaker_id = row["speaker_id"].strip()
            try:
                enrol_index = int(row["enrol_index"])
            except ValueError as error:
                raise ValueError(f"Invalid enrol_index at {path}:{row_number}") from error
            if not file_id or not speaker_id or enrol_index < 0:
                raise ValueError(f"Invalid enrolment reference at {path}:{row_number}")
            if file_id in seen_file_ids:
                raise ValueError(f"Duplicate development enrolment file ID: {file_id}")
            seen_file_ids.add(file_id)
            grouped[speaker_id].append((enrol_index, file_id))

    for speaker_id, references in grouped.items():
        references.sort(key=lambda reference: reference[0])
        indices = [index for index, _ in references]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate enrol_index for speaker {speaker_id}")
        if len(references) < max(ENROLMENT_SIZES):
            raise ValueError(
                f"Speaker {speaker_id} has fewer than {max(ENROLMENT_SIZES)} enrolment utterances"
            )
    if not grouped:
        raise ValueError(f"No development enrolment speakers found in {path}")
    return dict(grouped)


def _load_selected_enrolment_features(
    references: dict[str, list[tuple[int, str]]],
    enrolment_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    if enrolment_size not in ENROLMENT_SIZES:
        raise ValueError(f"Unsupported enrolment size: {enrolment_size}")
    feature_root = PROJECT_ROOT / "features_output" / "dev" / "features"
    speaker_features = {}
    speaker_files = {}
    loaded_features: dict[str, torch.Tensor] = {}
    for speaker_id, speaker_references in references.items():
        selected = sorted(speaker_references)[:enrolment_size]
        sequences = []
        file_ids = []
        for enrol_index, file_id in selected:
            path = feature_root / f"{file_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Development enrolment feature {file_id} (index {enrol_index}) is missing: {path}"
                )
            if file_id not in loaded_features:
                features = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(features, torch.Tensor):
                    raise TypeError(f"Feature file is not a tensor: {path}")
                if (
                    features.ndim != 2
                    or features.shape[1] != 402
                    or features.shape[0] == 0
                    or not features.is_floating_point()
                    or not torch.isfinite(features).all()
                ):
                    raise ValueError(f"Invalid Task 1 feature tensor: {path}")
                loaded_features[file_id] = features.to(torch.float64)
            sequences.append(loaded_features[file_id])
            file_ids.append(file_id)
        speaker_features[speaker_id] = torch.cat(sequences, dim=0).contiguous()
        speaker_files[speaker_id] = file_ids
    return speaker_features, speaker_files


def _load_task4_results() -> tuple[list[dict[str, str]], dict[int, dict[str, str]], dict[str, str]]:
    if not TASK4_COMPARISON.is_file():
        raise FileNotFoundError(f"Task 4 comparison table is missing: {TASK4_COMPARISON}")
    with TASK4_COMPARISON.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    relevance = {
        int(row["relevance_factor"]): row
        for row in rows
        if row["method"] == "MAP" and row["relevance_factor"]
    }
    if set(relevance) != set(RELEVANCE_FACTORS):
        raise ValueError("Task 4 results do not contain all required MAP factors")
    independent_rows = [row for row in rows if row["method"] == "independent_gmm"]
    if len(independent_rows) != 1:
        raise ValueError("Task 4 results must contain one independent-GMM baseline")
    independent = independent_rows[0]
    return rows, relevance, independent


def _metric_row(
    ablation: str,
    setting: str,
    changed: str,
    fixed: str,
    metrics: dict[str, float | int],
    source: str,
) -> dict[str, object]:
    return {
        "ablation": ablation,
        "setting": setting,
        "changed_setting": changed,
        "fixed_settings": fixed,
        "eer": metrics["eer"],
        "far_at_eer": metrics["far_at_threshold"],
        "frr_at_eer": metrics["frr_at_threshold"],
        "threshold_at_eer": metrics["threshold"],
        "total_trials": metrics["total_trials"],
        "genuine_trials": metrics["genuine_trials"],
        "impostor_trials": metrics["impostor_trials"],
        "source": source,
    }


def _task4_metric_row(
    ablation: str,
    setting: str,
    changed: str,
    fixed: str,
    row: dict[str, str],
) -> dict[str, object]:
    result = {
        "ablation": ablation,
        "setting": setting,
        "changed_setting": changed,
        "fixed_settings": fixed,
        "eer": float(row["eer"]),
        "far_at_eer": float(row["far_at_eer"]),
        "frr_at_eer": float(row["frr_at_eer"]),
        "threshold_at_eer": float(row["threshold_at_eer"]),
        "total_trials": int(row["total_trials"]),
        "genuine_trials": int(row["target_trials"]),
        "impostor_trials": int(row["nontarget_trials"]),
        "source": "reused Task 4 development comparison",
    }
    if ablation == "relevance_factor":
        result["relevance_factor"] = int(row["relevance_factor"])
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty Task 6 table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    relevance_rows: list[dict[str, object]],
    size_rows: list[dict[str, object]],
    model_rows: list[dict[str, object]],
    speaker_count: int,
    trial_count: int,
) -> None:
    lines = [
        "# Task 6 — Analysis and Ablation",
        "",
        "All comparisons use the same 1,824 labeled development trials, with 196 genuine and 1,628 impostor trials. Evaluation trials, labels, and scores were not read.",
        "",
        "## Relevance factor versus EER",
        "",
        "Reused the Task 4 development results for r=4, 8, 16, and 32. The changed setting is the MAP relevance factor; the UBM, all three enrolment utterances, feature representation, and trial set are fixed.",
        "",
        "| r | Development EER |",
        "|---:|---:|",
    ]
    lines.extend(
        f"| {row['setting']} | {float(row['eer']):.6f} |"
        for row in relevance_rows
    )
    lines.extend(
        [
            "",
            "EER decreases over the tested range as the relevance factor increases. In this experiment, stronger shrinkage toward the shared UBM produced better development verification than allowing the limited per-speaker enrolment evidence to move means more freely. This is an observed trend only within r=4–32.",
            "",
            "## Enrolment size versus EER",
            "",
            f"Re-adapted {speaker_count} development speakers with the unchanged 64-component diagonal UBM and r=32, selecting the first 1, 2, or 3 recordings per speaker after sorting by `enrol_index`. The changed setting is enrolment utterance count; MAP rule, UBM, features, test trials, and scoring are fixed. Every setting was scored against the same {trial_count} development trials.",
            "",
            "| Enrolment utterances | Development EER |",
            "|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['setting']} | {float(row['eer']):.6f} |"
        for row in size_rows
    )
    lines.extend(
        [
            "",
            "More enrolment utterances provide additional speaker evidence for the adapted means. The measured EERs show the actual effect on this development set; the three-utterance result should match Task 4's r=32 result because it uses the same recordings and pipeline.",
            "",
            "## MAP versus independent speaker GMM",
            "",
            "Reused the Task 4 comparison on the same development speakers, all three enrolment recordings, features, and trials. Only the speaker-model estimation method differs.",
            "",
            "| Method | Development EER |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| {row['setting']} | {float(row['eer']):.6f} |"
        for row in model_rows
    )
    lines.extend(
        [
            "",
            "The MAP model reuses the background UBM and adapts means only, while the independent model estimates its own GMM parameters from each speaker's small enrolment set. The Task 4 results show how these approaches compare under that data constraint.",
            "",
            "## Artifacts",
            "",
            "- `tables/comparison.csv`: all nine comparison rows and fixed/changed-setting descriptions.",
            "- `tables/relevance_factor.csv`, `tables/enrolment_size.csv`, and `tables/map_vs_independent_gmm.csv`: focused comparison tables.",
            "- `plots/relevance_factor_vs_eer.svg`, `plots/enrolment_size_vs_eer.svg`, and `plots/map_vs_independent_gmm.svg`.",
            "- Relevance-factor and model-method results were reused from `results/task4/tables/development_comparison.csv`; only the enrolment-size ablation was newly run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all() -> None:
    tables_dir = RESULTS_DIR / "tables"
    plots_dir = RESULTS_DIR / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, task4_relevance, task4_independent = _load_task4_results()
    trials, test_features = load_trials_and_features(PROJECT_ROOT, "dev")
    ubm = load_ubm(UBM_PATH)
    enrolment_references = _read_dev_enrolments()

    relevance_rows = [
        _task4_metric_row(
            "relevance_factor",
            f"r={factor}",
            "MAP relevance factor",
            "64-component diagonal UBM; 3 enrolment utterances; Task 1 features; dev_trials.csv",
            task4_relevance[factor],
        )
        for factor in RELEVANCE_FACTORS
    ]
    if any(int(row["total_trials"]) != len(trials) for row in relevance_rows):
        raise ValueError("Reused Task 4 relevance results use a different trial count")
    reference_counts = {
        int(row["genuine_trials"]): int(row["impostor_trials"])
        for row in relevance_rows
    }
    actual_counts = (
        sum(trial.label == 1 for trial in trials),
        sum(trial.label == 0 for trial in trials),
    )
    if reference_counts != {actual_counts[0]: actual_counts[1]}:
        raise ValueError("Reused Task 4 metrics do not match current development labels")

    size_rows = []
    for enrolment_size in ENROLMENT_SIZES:
        speaker_features, speaker_files = _load_selected_enrolment_features(
            enrolment_references,
            enrolment_size,
        )
        if set(speaker_features) != set(enrolment_references):
            raise RuntimeError("Not every development speaker received enrolment features")
        models: dict[str, GaussianMixture] = {}
        for speaker_id, features in speaker_features.items():
            adapted = adapt_means_map(
                features,
                ubm,
                SELECTED_RELEVANCE_FACTOR,
                occupancy_epsilon=OCCUPANCY_EPSILON,
                chunk_size=STATISTICS_CHUNK_SIZE,
            )
            if not torch.equal(adapted.model.weights, ubm.weights) or not torch.equal(
                adapted.model.covariances,
                ubm.covariances,
            ):
                raise RuntimeError("Mean-only MAP adaptation changed UBM parameters")
            models[speaker_id] = adapted.model
        scored = score_trials(
            trials,
            test_features,
            models,
            ubm,
            SCORING_CHUNK_SIZE,
        )
        metrics, _ = development_curve(scored)
        row = _metric_row(
            "enrolment_size",
            str(enrolment_size),
            "Number of enrolment utterances per speaker (selected by enrol_index)",
            "r=32; 64-component diagonal UBM; Task 1 features; same dev trials and LLR scorer",
            metrics,
            "new Task 6 development run",
        )
        row["enrolment_utterances"] = enrolment_size
        row["speaker_count"] = len(models)
        row["mean_pooled_frames"] = float(
            torch.tensor(
                [features.shape[0] for features in speaker_features.values()],
                dtype=torch.float64,
            ).mean().item()
        )
        row["enrolment_file_count"] = sum(len(file_ids) for file_ids in speaker_files.values())
        size_rows.append(row)
        print(
            f"Enrolment size {enrolment_size}: EER={float(metrics['eer']):.6f}, "
            f"speakers={len(models)}"
        )

    three_utterance_eer = float(size_rows[-1]["eer"])
    task4_r32_eer = float(task4_relevance[SELECTED_RELEVANCE_FACTOR]["eer"])
    if not math.isclose(three_utterance_eer, task4_r32_eer, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(
            "Three-utterance r=32 EER differs from the existing Task 4 result"
        )

    model_rows = [
        _task4_metric_row(
            "model_comparison",
            "MAP mean-only (r=32)",
            "Speaker model estimation method",
            "97 development speakers; 3 enrolment utterances; 64-component diagonal UBM; same features and trials",
            task4_relevance[SELECTED_RELEVANCE_FACTOR],
        ),
        _task4_metric_row(
            "model_comparison",
            "Independent diagonal GMM (64 components)",
            "Speaker model estimation method",
            "97 development speakers; 3 enrolment utterances; 64 components; same features and trials",
            task4_independent,
        ),
    ]

    all_rows = relevance_rows + size_rows + model_rows
    _write_csv(tables_dir / "relevance_factor.csv", relevance_rows)
    _write_csv(tables_dir / "enrolment_size.csv", size_rows)
    _write_csv(tables_dir / "map_vs_independent_gmm.csv", model_rows)
    _write_csv(tables_dir / "comparison.csv", all_rows)
    plot_relevance_factor(
        relevance_rows,
        plots_dir / "relevance_factor_vs_eer.svg",
    )
    plot_enrolment_size(
        size_rows,
        plots_dir / "enrolment_size_vs_eer.svg",
    )
    plot_model_comparison(
        model_rows,
        plots_dir / "map_vs_independent_gmm.svg",
    )
    _write_summary(
        RESULTS_DIR / "summary.md",
        relevance_rows,
        size_rows,
        model_rows,
        len(enrolment_references),
        len(trials),
    )
    print(
        f"Task 6 complete: {len(trials)} development trials; "
        f"{len(enrolment_references)} speakers; outputs in {RESULTS_DIR}"
    )


if __name__ == "__main__":
    run_all()
