"""Validated loading and deterministic preparation of Task 1 background features."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch


FEATURE_DIMENSION = 402


@dataclass(frozen=True)
class BackgroundRecording:
    """One feature sequence with its original background speaker and file ID."""

    file_id: str
    speaker_id: str
    features: torch.Tensor


def load_background_recordings(
    project_root: Path,
) -> list[BackgroundRecording]:
    """Load and validate only the background feature tensors and their speaker IDs."""
    split_root = project_root / "features_output" / "background"
    index_path = split_root / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Task 1 background index is missing: {index_path}")

    recordings = []
    seen_ids = set()
    with index_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"audio_id", "feature_path", "reference_json"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Background index must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            file_id = row["audio_id"]
            if file_id in seen_ids:
                raise ValueError(f"Repeated audio ID {file_id} in background index")
            seen_ids.add(file_id)
            reference = json.loads(row["reference_json"])
            speaker_id = reference.get("speaker_id")
            if not isinstance(speaker_id, str) or not speaker_id:
                raise ValueError(
                    f"Missing speaker_id in background index row {row_number}"
                )
            relative_path = Path(row["feature_path"].replace("\\", "/"))
            tensor_path = split_root / relative_path
            if not tensor_path.is_file():
                raise FileNotFoundError(
                    f"Feature tensor for {file_id} is missing: {tensor_path}"
                )
            features = torch.load(
                tensor_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(features, torch.Tensor):
                raise TypeError(f"Feature file is not a tensor: {tensor_path}")
            if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
                raise ValueError(
                    f"Expected {file_id} features shaped [frames, {FEATURE_DIMENSION}], "
                    f"got {tuple(features.shape)}"
                )
            if features.shape[0] == 0:
                raise ValueError(f"Background feature tensor is empty: {tensor_path}")
            if not torch.is_floating_point(features) or not torch.isfinite(features).all():
                raise ValueError(f"Invalid values in background tensor: {tensor_path}")
            recordings.append(
                BackgroundRecording(file_id, speaker_id, features.to(torch.float64))
            )
    if not recordings:
        raise ValueError("No background feature recordings were found")
    return recordings


def split_speakers(
    recordings: list[BackgroundRecording],
    heldout_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Return a deterministic disjoint partition of the speaker IDs."""
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be between zero and one")
    speaker_ids = sorted({recording.speaker_id for recording in recordings})
    if len(speaker_ids) < 2:
        raise ValueError("At least two speakers are needed for a held-out split")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(len(speaker_ids), generator=generator).tolist()
    heldout_count = min(
        len(speaker_ids) - 1,
        max(1, round(len(speaker_ids) * heldout_fraction)),
    )
    heldout = {speaker_ids[index] for index in permutation[:heldout_count]}
    fitting = set(speaker_ids) - heldout
    return fitting, heldout


def observations_for_speakers(
    recordings: list[BackgroundRecording],
    speaker_ids: set[str],
) -> torch.Tensor:
    """Concatenate all feature frames belonging to the requested speakers."""
    sequences = [
        recording.features
        for recording in recordings
        if recording.speaker_id in speaker_ids
    ]
    if not sequences:
        raise ValueError("No feature observations belong to the requested speakers")
    return torch.cat(sequences, dim=0)


def sample_per_speaker(
    recordings: list[BackgroundRecording],
    speaker_ids: set[str],
    max_frames_per_speaker: int,
    seed: int,
) -> torch.Tensor:
    """Select a reproducible, speaker-balanced subset of fitting feature frames."""
    if max_frames_per_speaker < 1:
        raise ValueError("max_frames_per_speaker must be positive")
    by_speaker: dict[str, list[torch.Tensor]] = {}
    for recording in recordings:
        if recording.speaker_id in speaker_ids:
            by_speaker.setdefault(recording.speaker_id, []).append(recording.features)

    selected = []
    for speaker_number, speaker_id in enumerate(sorted(by_speaker)):
        speaker_features = torch.cat(by_speaker[speaker_id], dim=0)
        count = min(max_frames_per_speaker, speaker_features.shape[0])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + speaker_number)
        indices = torch.randperm(speaker_features.shape[0], generator=generator)[:count]
        selected.append(speaker_features[indices])
    if not selected:
        raise ValueError("No fitting speakers were selected for frame sampling")
    return torch.cat(selected, dim=0).contiguous()
