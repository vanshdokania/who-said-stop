"""Score development/evaluation trials and create the Task 5 submission."""

import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.task5.data import Trial, load_trials_and_features
from experiments.task5.plots import plot_roc
from experiments.task5.scoring import (
    development_curve,
    load_map_models,
    load_ubm,
    operating_rates,
    score_trials,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "task5"
TASK4_METADATA = PROJECT_ROOT / "results" / "task4" / "tables" / "task4_metadata.json"
UBM_PATH = PROJECT_ROOT / "results" / "task3" / "models" / "ubm_final.pt"
CHUNK_SIZE = 1024


def _validate_claims(
    trials: list[Trial],
    models: dict[str, object],
    split: str,
) -> None:
    missing = sorted({trial.claimed_id for trial in trials} - set(models))
    if missing:
        raise ValueError(
            f"{split} trials reference claimed speakers without MAP models: "
            f"{missing[:10]}"
        )


def _validate_scored_rows(
    trials: list[Trial],
    scored: list[dict[str, object]],
    split: str,
) -> None:
    if len(scored) != len(trials):
        raise RuntimeError(
            f"{split}: scored {len(scored)} trials, expected {len(trials)}"
        )
    expected_ids = [trial.trial_id for trial in trials]
    actual_ids = [str(row["trial_id"]) for row in scored]
    if actual_ids != expected_ids:
        raise RuntimeError(f"{split}: scored trial IDs/order do not match manifest")
    for trial, row in zip(trials, scored):
        if row["claimed_id"] != trial.claimed_id:
            raise RuntimeError(f"{split}: claimed speaker mapping changed for {trial.trial_id}")
        score = float(row["score"])
        if not math.isfinite(score):
            raise FloatingPointError(f"{split}: non-finite score for {trial.trial_id}")


def _write_trial_scores(
    path: Path,
    rows: list[dict[str, object]],
    include_label: bool,
) -> None:
    fields = ["trial_id", "claimed_id", "test_file"]
    if include_label:
        fields.append("label")
    fields.append("score")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_submission(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["trial_id", "score"])
        writer.writeheader()
        for row in rows:
            score = float(row["score"])
            if not math.isfinite(score):
                raise FloatingPointError(f"Non-finite submission score: {row['trial_id']}")
            writer.writerow(
                {"trial_id": row["trial_id"], "score": format(score, ".17g")}
            )
    temporary_path.replace(path)


def _validate_submission(
    path: Path,
    eval_trials: list[Trial],
) -> None:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["trial_id", "score"]:
            raise ValueError("submission.csv must have exactly trial_id,score columns")
        rows = list(reader)
    expected_ids = [trial.trial_id for trial in eval_trials]
    actual_ids = [row["trial_id"] for row in rows]
    if len(rows) != len(eval_trials) or actual_ids != expected_ids:
        raise ValueError("submission.csv row count, trial IDs, or trial order is incorrect")
    for row in rows:
        try:
            score = float(row["score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Submission score is not numeric for {row['trial_id']}") from error
        if not math.isfinite(score):
            raise ValueError(f"Submission score is not finite for {row['trial_id']}")


def _write_summary(
    path: Path,
    metrics: dict[str, float | int],
    relevance_factor: int,
    eval_count: int,
    submission_path: Path,
) -> None:
    lines = [
        "# Task 5 — Verification and Scoring",
        "",
        "## Scoring method",
        "",
        "For each trial, Task 1 feature frames are scored under the claimed speaker's Task 4 mean-only MAP model and the fixed Task 3 UBM. The PyTorch GMM implementation computes each mixture log likelihood in the log domain with `logsumexp`; the trial score is the speaker model's mean frame log likelihood minus the UBM's mean frame log likelihood. Higher values favour the claimed identity.",
        f"The selected Task 4 relevance factor was r={relevance_factor}. No UBM training or MAP adaptation was repeated.",
        "",
        "## Development evaluation",
        "",
        f"- Trials: {metrics['total_trials']} total; {metrics['genuine_trials']} genuine and {metrics['impostor_trials']} impostor.",
        f"- EER: {float(metrics['eer']):.6f}.",
        f"- Operating threshold: **{float(metrics['threshold']):.17g}**.",
        f"- FAR at threshold: {float(metrics['far_at_threshold']):.6f}.",
        f"- FRR at threshold: {float(metrics['frr_at_threshold']):.6f}.",
        "- The threshold is the development ROC point closest to FAR=FRR, appropriate as a balanced-error operating point when no application-specific error costs or target operating rate are specified. Accept when score >= threshold; otherwise reject. A higher threshold typically reduces false acceptances while increasing false rejections, and vice versa.",
        "- A ranking score is a continuous LLR used to compare trial evidence; it is not itself a probability. A threshold turns the score into an accept/reject decision. A calibrated probability requires a separate calibration procedure and representative calibration data; none was applied here.",
        "",
        "## Evaluation and submission",
        "",
        f"- Scored all {eval_count} evaluation trials using the matching evaluation speaker model for each `claimed_id`. Evaluation labels/results were not used for model selection.",
        f"- Submission: `{submission_path.relative_to(PROJECT_ROOT)}` with exactly `trial_id,score`, manifest order preserved, and all values finite and numeric.",
        "- These are continuous scores, not binary predictions.",
        "",
        "## Artifacts",
        "",
        "- `tables/development_trial_scores.csv`: one development LLR per trial, with labels.",
        "- `tables/evaluation_trial_scores.csv`: one evaluation LLR per trial.",
        "- `tables/development_roc.csv`: threshold, FAR, FRR, and TAR operating points.",
        "- `tables/development_metrics.json`: EER, selected threshold, and trial counts.",
        "- `plots/development_roc.svg`: ROC curve with the selected operating point.",
        "- `logs/task5.log`: run and validation counts.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all() -> None:
    tables_dir = RESULTS_DIR / "tables"
    plots_dir = RESULTS_DIR / "plots"
    logs_dir = RESULTS_DIR / "logs"
    for directory in (tables_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not TASK4_METADATA.is_file():
        raise FileNotFoundError(f"Task 4 metadata is missing: {TASK4_METADATA}")
    with TASK4_METADATA.open("r", encoding="utf-8") as file:
        task4_metadata = json.load(file)
    relevance_factor = task4_metadata.get("selected_relevance_factor")
    if not isinstance(relevance_factor, int):
        raise ValueError("Task 4 metadata has no valid selected relevance factor")
    dev_model_path = (
        PROJECT_ROOT
        / "results"
        / "task4"
        / "models"
        / f"development_map_r{relevance_factor}.pt"
    )
    eval_model_path = (
        PROJECT_ROOT
        / "results"
        / "task4"
        / "models"
        / f"evaluation_map_r{relevance_factor}.pt"
    )

    ubm = load_ubm(UBM_PATH)
    dev_models, dev_relevance = load_map_models(dev_model_path, "dev", ubm)
    eval_models, eval_relevance = load_map_models(eval_model_path, "eval", ubm)
    if dev_relevance != relevance_factor or eval_relevance != relevance_factor:
        raise ValueError("Task 4 development/evaluation MAP factors do not match selection")

    dev_trials, dev_features = load_trials_and_features(PROJECT_ROOT, "dev")
    eval_trials, eval_features = load_trials_and_features(PROJECT_ROOT, "eval")
    _validate_claims(dev_trials, dev_models, "Development")
    _validate_claims(eval_trials, eval_models, "Evaluation")
    dev_scored = score_trials(dev_trials, dev_features, dev_models, ubm, CHUNK_SIZE)
    eval_scored = score_trials(eval_trials, eval_features, eval_models, ubm, CHUNK_SIZE)
    _validate_scored_rows(dev_trials, dev_scored, "Development")
    _validate_scored_rows(eval_trials, eval_scored, "Evaluation")

    metrics, roc_rows = development_curve(dev_scored)
    far, frr = operating_rates(dev_scored, float(metrics["threshold"]))
    if (
        not math.isclose(far, float(metrics["far_at_threshold"]), abs_tol=1e-12)
        or not math.isclose(frr, float(metrics["frr_at_threshold"]), abs_tol=1e-12)
    ):
        raise RuntimeError("Selected ROC threshold does not reproduce reported FAR/FRR")

    dev_scores_path = tables_dir / "development_trial_scores.csv"
    eval_scores_path = tables_dir / "evaluation_trial_scores.csv"
    _write_trial_scores(dev_scores_path, dev_scored, include_label=True)
    _write_trial_scores(eval_scores_path, eval_scored, include_label=False)
    with (tables_dir / "development_roc.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["threshold", "far", "frr", "tar"],
        )
        writer.writeheader()
        writer.writerows(roc_rows)
    metrics["far_at_threshold"] = far
    metrics["frr_at_threshold"] = frr
    metrics_path = tables_dir / "development_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
        file.write("\n")

    roc_path = plots_dir / "development_roc.svg"
    plot_roc(roc_rows, float(metrics["threshold"]), roc_path)
    submission_path = PROJECT_ROOT / "submission.csv"
    _write_submission(submission_path, eval_scored)
    _validate_submission(submission_path, eval_trials)

    log_lines = [
        "Task 5 scoring completed.",
        f"UBM: {UBM_PATH.relative_to(PROJECT_ROOT)}",
        f"Development MAP models: {dev_model_path.relative_to(PROJECT_ROOT)}",
        f"Evaluation MAP models: {eval_model_path.relative_to(PROJECT_ROOT)}",
        f"Selected relevance factor: {relevance_factor}",
        f"Development trials scored: {len(dev_scored)}",
        f"Development unique test files: {len(dev_features)}",
        f"Genuine/impostor trials: {metrics['genuine_trials']}/{metrics['impostor_trials']}",
        f"Development EER: {float(metrics['eer']):.8f}",
        f"Threshold/FAR/FRR: {float(metrics['threshold']):.17g}/{far:.8f}/{frr:.8f}",
        f"Evaluation trials scored: {len(eval_scored)}",
        f"Evaluation unique test files: {len(eval_features)}",
        "Submission columns validated: trial_id,score",
        f"Submission rows and order validated: {len(eval_trials)}",
        f"Submission file: {submission_path.relative_to(PROJECT_ROOT)}",
    ]
    (logs_dir / "task5.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    _write_summary(
        RESULTS_DIR / "summary.md",
        metrics,
        relevance_factor,
        len(eval_scored),
        submission_path,
    )
    print("\n".join(log_lines))


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
