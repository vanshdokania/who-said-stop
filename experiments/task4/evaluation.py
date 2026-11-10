"""Development-only claimed-speaker likelihood-ratio evaluation."""

from collections import defaultdict

import torch

from experiments.task4.data import DevelopmentTrial
from src.gmm.gmm import GaussianMixture
from src.gmm.likelihood import component_log_probabilities


def _average_log_likelihood(
    features: torch.Tensor,
    model: GaussianMixture,
    chunk_size: int,
) -> float:
    if features.shape[0] == 0:
        raise ValueError("Cannot score empty test features")
    component_scores = component_log_probabilities(features, model, chunk_size)
    return float(torch.logsumexp(component_scores, dim=1).mean().item())


def score_claimed_speaker_trials(
    trials: list[DevelopmentTrial],
    feature_cache: dict[str, torch.Tensor],
    models: dict[str, GaussianMixture],
    ubm: GaussianMixture,
    chunk_size: int = 1024,
) -> list[dict[str, object]]:
    """Score only the claims in dev_trials with speaker-vs-UBM log likelihood ratios."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for trial_index, trial in enumerate(trials):
        if trial.claimed_id not in models:
            raise ValueError(f"No adapted model for claimed speaker {trial.claimed_id}")
        if trial.test_file not in feature_cache:
            raise ValueError(f"No development features for {trial.test_file}")
        grouped[trial.claimed_id].append(trial_index)

    scores = [0.0] * len(trials)
    ubm_cache = {
        test_file: _average_log_likelihood(features, ubm, chunk_size)
        for test_file, features in feature_cache.items()
    }
    for speaker_id, indices in grouped.items():
        model = models[speaker_id]
        unique_files = list(dict.fromkeys(trials[index].test_file for index in indices))
        total_frames = sum(feature_cache[test_file].shape[0] for test_file in unique_files)
        pooled = torch.cat([feature_cache[test_file] for test_file in unique_files], dim=0)
        component_scores = component_log_probabilities(pooled, model, chunk_size)
        frame_log_likelihoods = torch.logsumexp(component_scores, dim=1)
        cursor = 0
        speaker_scores = {}
        for test_file in unique_files:
            count = feature_cache[test_file].shape[0]
            avg = float(frame_log_likelihoods[cursor : cursor + count].mean().item())
            speaker_scores[test_file] = avg - ubm_cache[test_file]
            cursor += count
        if cursor != total_frames:
            raise RuntimeError("Internal test-frame accounting mismatch")
        for trial_index in indices:
            scores[trial_index] = speaker_scores[trials[trial_index].test_file]

    return [
        {
            "trial_id": trial.trial_id,
            "claimed_id": trial.claimed_id,
            "test_file": trial.test_file,
            "label": trial.label,
            "score": scores[index],
        }
        for index, trial in enumerate(trials)
    ]


def verification_metrics(
    scored_trials: list[dict[str, object]],
) -> dict[str, float | int]:
    """Calculate EER, FAR, and FRR at the closest operating point."""
    if not scored_trials:
        raise ValueError("No trials were supplied for verification metrics")
    labels = torch.tensor(
        [int(row["label"]) for row in scored_trials],
        dtype=torch.int64,
    )
    scores = torch.tensor(
        [float(row["score"]) for row in scored_trials],
        dtype=torch.float64,
    )
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Trial scores contain NaN or infinity")
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError("EER requires both target and non-target trials")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = torch.cat(
        (
            torch.nonzero(sorted_scores[:-1] != sorted_scores[1:]).flatten(),
            torch.tensor([len(sorted_scores) - 1]),
        )
    )
    cumulative_targets = torch.cumsum(sorted_labels == 1, dim=0)[group_ends]
    cumulative_nontargets = torch.cumsum(sorted_labels == 0, dim=0)[group_ends]
    far = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            cumulative_nontargets.to(torch.float64) / negatives,
        )
    )
    frr = torch.cat(
        (
            torch.ones(1, dtype=torch.float64),
            1.0 - cumulative_targets.to(torch.float64) / positives,
        )
    )
    thresholds = torch.cat(
        (
            torch.nextafter(
                sorted_scores[:1],
                torch.tensor(float("inf"), dtype=sorted_scores.dtype),
            ),
            sorted_scores[group_ends],
        )
    )
    best_index = int((far - frr).abs().argmin().item())
    return {
        "eer": float(((far[best_index] + frr[best_index]) / 2).item()),
        "far_at_eer": float(far[best_index].item()),
        "frr_at_eer": float(frr[best_index].item()),
        "threshold_at_eer": float(thresholds[best_index].item()),
        "target_trials": positives,
        "nontarget_trials": negatives,
        "total_trials": len(scored_trials),
    }
