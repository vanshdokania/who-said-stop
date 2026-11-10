"""Load enrolment and development trial features for Task 4."""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch


FEATURE_DIMENSION = 402


@dataclass(frozen=True)
class EnrolmentSpeaker:
    """Enrolment feature pool and source files for one speaker."""

    speaker_id: str
    enrolment_indices: tuple[int, ...]
    file_ids: tuple[str, ...]
    features: torch.Tensor


@dataclass(frozen=True)
class DevelopmentTrial:
    """A development claim, test utterance, and ground-truth label."""

    trial_id: str
    claimed_id: str
    test_file: str
    label: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Task 4 manifest is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader)


def _load_feature(
    feature_root: Path,
    file_id: str,
) -> torch.Tensor:
    path = feature_root / "features" / f"{file_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Feature tensor for {file_id} is missing: {path}")
    features = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Feature file is not a tensor: {path}")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Expected {file_id} features shaped [frames, {FEATURE_DIMENSION}], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise ValueError(f"Feature tensor is empty: {path}")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError(f"Feature tensor has invalid dtype or values: {path}")
    return features.to(torch.float64)


def load_enrolment_speakers(
    project_root: Path,
    split: str,
) -> dict[str, EnrolmentSpeaker]:
    """Pool all listed enrolment recordings by speaker in enrol_index order."""
    if split not in ("dev", "eval"):
        raise ValueError("split must be 'dev' or 'eval'")
    manifest_path = project_root / "data" / f"{split}_enrol.csv"
    feature_root = project_root / "features_output" / split
    rows = _read_csv(manifest_path)
    required = {"file_id", "speaker_id", "enrol_index"}
    if not required.issubset(rows[0] if rows else {}):
        raise ValueError(f"{manifest_path} must contain columns {sorted(required)}")

    grouped: dict[str, list[tuple[int, str]]] = {}
    seen_files = set()
    for row_number, row in enumerate(rows, start=2):
        file_id = row["file_id"].strip()
        speaker_id = row["speaker_id"].strip()
        try:
            enrol_index = int(row["enrol_index"])
        except ValueError as error:
            raise ValueError(
                f"Invalid enrol_index in {manifest_path} row {row_number}"
            ) from error
        if not file_id or not speaker_id or enrol_index < 0:
            raise ValueError(f"Invalid enrolment reference in {manifest_path}:{row_number}")
        if file_id in seen_files:
            raise ValueError(f"Duplicate enrolment file ID: {file_id}")
        seen_files.add(file_id)
        grouped.setdefault(speaker_id, []).append((enrol_index, file_id))

    speakers = {}
    for speaker_id, references in grouped.items():
        references.sort(key=lambda reference: reference[0])
        indices = [reference[0] for reference in references]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate enrol_index for speaker {speaker_id}")
        file_ids = tuple(reference[1] for reference in references)
        sequences = [
            _load_feature(feature_root, file_id)
            for file_id in file_ids
        ]
        speakers[speaker_id] = EnrolmentSpeaker(
            speaker_id=speaker_id,
            enrolment_indices=tuple(indices),
            file_ids=file_ids,
            features=torch.cat(sequences, dim=0).contiguous(),
        )
    if not speakers:
        raise ValueError(f"No enrolment speakers found in {manifest_path}")
    return speakers


def load_development_trials(
    project_root: Path,
) -> tuple[list[DevelopmentTrial], dict[str, torch.Tensor]]:
    """Load labeled development trials and cache their test-file features."""
    rows = _read_csv(project_root / "data" / "dev_trials.csv")
    required = {"trial_id", "claimed_id", "test_file", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"dev_trials.csv must contain columns {sorted(required)}"
        )
    feature_root = project_root / "features_output" / "dev"
    trials = []
    feature_cache = {}
    seen_trial_ids = set()
    for row_number, row in enumerate(rows, start=2):
        trial_id = row["trial_id"].strip()
        claimed_id = row["claimed_id"].strip()
        test_file = row["test_file"].strip()
        try:
            label = int(row["label"])
        except ValueError as error:
            raise ValueError(f"Invalid development label at row {row_number}") from error
        if not trial_id or not claimed_id or not test_file or label not in (0, 1):
            raise ValueError(f"Invalid development trial at row {row_number}")
        if trial_id in seen_trial_ids:
            raise ValueError(f"Duplicate development trial ID: {trial_id}")
        seen_trial_ids.add(trial_id)
        if test_file not in feature_cache:
            feature_cache[test_file] = _load_feature(feature_root, test_file)
        trials.append(
            DevelopmentTrial(trial_id, claimed_id, test_file, label)
        )
    return trials, feature_cache
