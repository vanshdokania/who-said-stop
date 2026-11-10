"""Load the exact Task 2 background split and deterministic frame sample."""

import csv
from pathlib import Path

import torch

from src.utils.data_loader import (
    BackgroundRecording,
    load_background_recordings,
    observations_for_speakers,
    sample_per_speaker,
)


def load_task2_split(
    project_root: Path,
    recordings: list[BackgroundRecording],
) -> tuple[set[str], set[str]]:
    """Read Task 2 split assignments and verify every file/speaker mapping."""
    split_path = project_root / "results" / "task2" / "tables" / "speaker_split.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Task 2 speaker split is missing: {split_path}")
    expected = {recording.file_id: recording.speaker_id for recording in recordings}
    fit_speakers: set[str] = set()
    heldout_speakers: set[str] = set()
    seen_files = set()
    with split_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"file_id", "speaker_id", "subset"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Task 2 split must contain {sorted(required_columns)}"
            )
        for row_number, row in enumerate(reader, start=2):
            file_id = row["file_id"]
            if file_id in seen_files:
                raise ValueError(f"Duplicate file ID in Task 2 split row {row_number}")
            seen_files.add(file_id)
            if expected.get(file_id) != row["speaker_id"]:
                raise ValueError(
                    f"Task 2 split speaker mapping disagrees for {file_id}"
                )
            if row["subset"] == "fitting":
                fit_speakers.add(row["speaker_id"])
            elif row["subset"] == "heldout":
                heldout_speakers.add(row["speaker_id"])
            else:
                raise ValueError(
                    f"Unknown subset in Task 2 split row {row_number}: {row['subset']}"
                )
    if seen_files != set(expected):
        raise ValueError("Task 2 split does not cover exactly the background recordings")
    if fit_speakers & heldout_speakers:
        raise ValueError("Task 2 speaker split contains fitting/held-out leakage")
    return fit_speakers, heldout_speakers


def load_task3_data(
    project_root: Path,
    max_frames_per_speaker: int = 16,
    frame_seed: int = 2027,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
    """Load fitting and held-out observations using Task 2's fixed procedure."""
    recordings = load_background_recordings(project_root)
    fitting_speakers, heldout_speakers = load_task2_split(project_root, recordings)
    fitting_recordings = [
        recording for recording in recordings
        if recording.speaker_id in fitting_speakers
    ]
    heldout_recordings = [
        recording for recording in recordings
        if recording.speaker_id in heldout_speakers
    ]
    fitting_data = sample_per_speaker(
        recordings,
        fitting_speakers,
        max_frames_per_speaker,
        frame_seed,
    )
    heldout_data = observations_for_speakers(recordings, heldout_speakers)
    if not torch.isfinite(fitting_data).all() or not torch.isfinite(heldout_data).all():
        raise FloatingPointError("Task 3 background data contains non-finite features")
    metadata: dict[str, int | str] = {
        "recording_count": len(recordings),
        "fitting_speaker_count": len(fitting_speakers),
        "heldout_speaker_count": len(heldout_speakers),
        "fitting_recording_count": len(fitting_recordings),
        "heldout_recording_count": len(heldout_recordings),
        "fitting_frames": fitting_data.shape[0],
        "heldout_frames": heldout_data.shape[0],
        "max_fit_frames_per_speaker": max_frames_per_speaker,
        "fit_frame_seed": frame_seed,
        "split_source": "results/task2/tables/speaker_split.csv",
    }
    return fitting_data, heldout_data, metadata
