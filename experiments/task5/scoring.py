"""Stable PyTorch speaker-versus-UBM scoring and development metrics."""

import math
from collections import defaultdict
from pathlib import Path

import torch

from experiments.task5.data import Trial
from src.gmm.gmm import GaussianMixture
from src.gmm.likelihood import average_log_likelihood, component_log_probabilities


UBM_FORMAT = "who_said_stop_gmm_ubm_v1"
MAP_FORMAT = "who_said_stop_map_mean_models_v1"


def load_ubm(path: Path) -> GaussianMixture:
    """Load and validate the saved Task 3 UBM without fitting it."""
    if not path.is_file():
        raise FileNotFoundError(f"Task 3 UBM checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != UBM_FORMAT:
        raise ValueError(f"Unsupported Task 3 UBM checkpoint: {path}")
    parameters = checkpoint.get("model")
    if not isinstance(parameters, dict):
        raise ValueError(f"Task 3 UBM checkpoint has no model parameters: {path}")
    model = GaussianMixture(
        weights=parameters["weights"].to(torch.float64),
        means=parameters["means"].to(torch.float64),
        covariances=parameters["covariances"].to(torch.float64),
        covariance_type=parameters["covariance_type"],
        reg_covar=float(parameters["reg_covar"]),
    )
    _validate_model(model, expected_components=64, expected_features=402)
    if model.covariance_type != "diagonal":
        raise ValueError("Task 5 expects the Task 3 diagonal UBM")
    if model.covariances.shape != model.means.shape:
        raise ValueError("Diagonal UBM covariances must have shape [K, D]")
    return model


def load_map_models(
    path: Path,
    expected_split: str,
    ubm: GaussianMixture,
) -> tuple[dict[str, GaussianMixture], int]:
    """Load saved mean-only MAP speakers with the unchanged UBM parameters."""
    if expected_split not in ("dev", "eval"):
        raise ValueError("expected_split must be 'dev' or 'eval'")
    if not path.is_file():
        raise FileNotFoundError(f"Task 4 MAP checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != MAP_FORMAT:
        raise ValueError(f"Unsupported Task 4 MAP checkpoint: {path}")
    if checkpoint.get("split") != expected_split:
        raise ValueError(
            f"Expected {expected_split} MAP models, got {checkpoint.get('split')!r}"
        )
    if checkpoint.get("adaptation") != "mean_only_map":
        raise ValueError("Task 5 requires mean-only MAP speaker models")
    if (
        checkpoint.get("covariance_type") != ubm.covariance_type
        or checkpoint.get("feature_dimension") != ubm.n_features
        or checkpoint.get("n_components") != ubm.n_components
    ):
        raise ValueError("Task 4 MAP checkpoint does not match the Task 3 UBM")

    weights = checkpoint["weights"].to(dtype=ubm.weights.dtype, device="cpu")
    covariances = checkpoint["covariances"].to(
        dtype=ubm.covariances.dtype,
        device="cpu",
    )
    if not torch.equal(weights, ubm.weights) or not torch.equal(
        covariances,
        ubm.covariances,
    ):
        raise ValueError("Saved MAP weights/covariances differ from the loaded UBM")
    if not math.isfinite(float(checkpoint.get("relevance_factor", float("nan")))):
        raise ValueError("MAP checkpoint has an invalid relevance factor")

    records = checkpoint.get("speakers")
    if not isinstance(records, dict) or not records:
        raise ValueError("Task 4 MAP checkpoint contains no speaker models")
    models = {}
    for speaker_id, record in records.items():
        if not isinstance(speaker_id, str) or not speaker_id or not isinstance(record, dict):
            raise ValueError("Task 4 MAP checkpoint contains an invalid speaker entry")
        means = record.get("means")
        if not isinstance(means, torch.Tensor):
            raise TypeError(f"Missing mean tensor for speaker {speaker_id}")
        model = GaussianMixture(
            weights=weights,
            means=means.to(dtype=ubm.means.dtype, device="cpu"),
            covariances=covariances,
            covariance_type=ubm.covariance_type,
            reg_covar=ubm.reg_covar,
        )
        _validate_model(model, ubm.n_components, ubm.n_features)
        models[speaker_id] = model
    return models, int(checkpoint["relevance_factor"])


def _validate_model(
    model: GaussianMixture,
    expected_components: int,
    expected_features: int,
) -> None:
    if model.weights.shape != (expected_components,):
        raise ValueError("GMM weights have an unexpected shape")
    if model.means.shape != (expected_components, expected_features):
        raise ValueError("GMM means have an unexpected shape")
    if not (
        torch.isfinite(model.weights).all()
        and torch.isfinite(model.means).all()
        and torch.isfinite(model.covariances).all()
    ):
        raise ValueError("GMM contains NaN or infinite parameters")
    if (model.weights <= 0).any() or (model.covariances <= 0).any():
        raise ValueError("GMM weights and covariance values must be positive")
    if not torch.isclose(
        model.weights.sum(),
        torch.ones((), dtype=model.weights.dtype),
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError("GMM weights must sum to one")


def score_trials(
    trials: list[Trial],
    feature_cache: dict[str, torch.Tensor],
    models: dict[str, GaussianMixture],
    ubm: GaussianMixture,
    chunk_size: int = 1024,
) -> list[dict[str, object]]:
    """Return one average per-frame LLR for every trial, preserving input order."""
    if not trials:
        raise ValueError("Cannot score an empty trial list")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, trial in enumerate(trials):
        if trial.claimed_id not in models:
            raise ValueError(f"No adapted model for claimed speaker {trial.claimed_id}")
        if trial.test_file not in feature_cache:
            raise ValueError(f"No cached features for test recording {trial.test_file}")
        features = feature_cache[trial.test_file]
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != ubm.n_features
            or not torch.isfinite(features).all()
        ):
            raise ValueError(f"Invalid test features for {trial.test_file}")
        grouped[trial.claimed_id].append(index)

    ubm_scores = {
        test_file: average_log_likelihood(features, ubm, chunk_size)
        for test_file, features in feature_cache.items()
    }
    scores = [0.0] * len(trials)
    for claimed_id, indices in grouped.items():
        unique_files = list(dict.fromkeys(trials[index].test_file for index in indices))
        pooled = torch.cat([feature_cache[file_id] for file_id in unique_files], dim=0)
        component_scores = component_log_probabilities(
            pooled,
            models[claimed_id],
            chunk_size,
        )
        frame_log_likelihoods = torch.logsumexp(component_scores, dim=1)
        cursor = 0
        pair_scores = {}
        for file_id in unique_files:
            frame_count = feature_cache[file_id].shape[0]
            speaker_average = float(
                frame_log_likelihoods[cursor : cursor + frame_count].mean().item()
            )
            pair_score = speaker_average - ubm_scores[file_id]
            if not math.isfinite(pair_score):
                raise FloatingPointError(
                    f"Non-finite LLR for claimed speaker {claimed_id}, file {file_id}"
                )
            pair_scores[file_id] = pair_score
            cursor += frame_count
        if cursor != pooled.shape[0]:
            raise RuntimeError("Internal test-frame accounting mismatch")
        for index in indices:
            scores[index] = pair_scores[trials[index].test_file]

    scored = []
    for trial, score in zip(trials, scores):
        row: dict[str, object] = {
            "trial_id": trial.trial_id,
            "claimed_id": trial.claimed_id,
            "test_file": trial.test_file,
            "score": score,
        }
        if trial.label is not None:
            row["label"] = trial.label
        scored.append(row)
    return scored


def development_curve(
    scored_trials: list[dict[str, object]],
) -> tuple[dict[str, float | int], list[dict[str, float]]]:
    """Build ROC operating points and select the point closest to equal error."""
    if not scored_trials:
        raise ValueError("No development trials were supplied")
    labels = torch.tensor(
        [int(row["label"]) for row in scored_trials],
        dtype=torch.int64,
    )
    scores = torch.tensor(
        [float(row["score"]) for row in scored_trials],
        dtype=torch.float64,
    )
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Development scores contain NaN or infinity")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Development labels must be binary 0/1")
    genuine_count = int((labels == 1).sum().item())
    impostor_count = int((labels == 0).sum().item())
    if genuine_count == 0 or impostor_count == 0:
        raise ValueError("ROC/EER requires genuine and impostor trials")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = torch.cat(
        (
            torch.nonzero(sorted_scores[:-1] != sorted_scores[1:]).flatten(),
            torch.tensor([len(sorted_scores) - 1]),
        )
    )
    cumulative_genuine = torch.cumsum(sorted_labels == 1, dim=0)[group_ends]
    cumulative_impostor = torch.cumsum(sorted_labels == 0, dim=0)[group_ends]
    far = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            cumulative_impostor.to(torch.float64) / impostor_count,
        )
    )
    frr = torch.cat(
        (
            torch.ones(1, dtype=torch.float64),
            1.0 - cumulative_genuine.to(torch.float64) / genuine_count,
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
    roc_rows = [
        {
            "threshold": float(thresholds[index].item()),
            "far": float(far[index].item()),
            "frr": float(frr[index].item()),
            "tar": float((1.0 - frr[index]).item()),
        }
        for index in range(len(thresholds))
    ]
    best_index = int((far - frr).abs().argmin().item())
    metrics: dict[str, float | int] = {
        "eer": float(((far[best_index] + frr[best_index]) / 2).item()),
        "far_at_threshold": float(far[best_index].item()),
        "frr_at_threshold": float(frr[best_index].item()),
        "threshold": float(thresholds[best_index].item()),
        "genuine_trials": genuine_count,
        "impostor_trials": impostor_count,
        "total_trials": len(scored_trials),
    }
    return metrics, roc_rows


def operating_rates(
    scored_trials: list[dict[str, object]],
    threshold: float,
) -> tuple[float, float]:
    """Calculate development FAR and FRR for the decision score >= threshold."""
    genuine = [row for row in scored_trials if int(row["label"]) == 1]
    impostor = [row for row in scored_trials if int(row["label"]) == 0]
    if not genuine or not impostor:
        raise ValueError("FAR and FRR require genuine and impostor trials")
    far = sum(float(row["score"]) >= threshold for row in impostor) / len(impostor)
    frr = sum(float(row["score"]) < threshold for row in genuine) / len(genuine)
    return far, frr
