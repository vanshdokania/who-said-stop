"""Create a reproducible speaker-disjoint split of background feature data."""

import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (
    BackgroundRecording,
    load_background_recordings,
    observations_for_speakers,
    sample_per_speaker,
    split_speakers,
)
from src.utils.logging_utils import write_json


HELDOUT_FRACTION = 0.20
SPLIT_SEED = 2026


def create_split(
    recordings: list[BackgroundRecording],
    output_dir: Path,
    heldout_fraction: float = HELDOUT_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[set[str], set[str]]:
    """Write a file-level split table whose assignments are speaker-disjoint."""
    fitting_speakers, heldout_speakers = split_speakers(
        recordings,
        heldout_fraction,
        seed,
    )
    if fitting_speakers & heldout_speakers:
        raise RuntimeError("Speaker split contains leakage")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "speaker_split.csv"
    with split_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("file_id", "speaker_id", "subset"))
        writer.writeheader()
        for recording in recordings:
            writer.writerow(
                {
                    "file_id": recording.file_id,
                    "speaker_id": recording.speaker_id,
                    "subset": (
                        "heldout"
                        if recording.speaker_id in heldout_speakers
                        else "fitting"
                    ),
                }
            )
    fitting_recordings = [
        recording for recording in recordings
        if recording.speaker_id in fitting_speakers
    ]
    heldout_recordings = [
        recording for recording in recordings
        if recording.speaker_id in heldout_speakers
    ]
    metadata = {
        "method": "torch.randperm over sorted unique background speaker IDs",
        "seed": seed,
        "heldout_fraction": heldout_fraction,
        "speaker_count": len(fitting_speakers) + len(heldout_speakers),
        "fitting_speaker_count": len(fitting_speakers),
        "heldout_speaker_count": len(heldout_speakers),
        "recording_count": len(recordings),
        "fitting_recording_count": len(fitting_recordings),
        "heldout_recording_count": len(heldout_recordings),
        "fitting_frame_count": sum(r.features.shape[0] for r in fitting_recordings),
        "heldout_frame_count": sum(r.features.shape[0] for r in heldout_recordings),
        "fitting_speaker_ids": sorted(fitting_speakers),
        "heldout_speaker_ids": sorted(heldout_speakers),
    }
    write_json(output_dir / "speaker_split.json", metadata)
    return fitting_speakers, heldout_speakers


def prepare_background_observations(
    project_root: Path,
    tables_dir: Path,
    heldout_fraction: float = HELDOUT_FRACTION,
    split_seed: int = SPLIT_SEED,
    max_fit_frames_per_speaker: int = 16,
    fit_frame_seed: int = 2027,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Load background tensors, create the fixed split, and prepare model inputs."""
    recordings = load_background_recordings(project_root)
    fitting_speakers, heldout_speakers = create_split(
        recordings,
        tables_dir,
        heldout_fraction=heldout_fraction,
        seed=split_seed,
    )
    if fitting_speakers & heldout_speakers:
        raise RuntimeError("Speaker leakage detected")
    fitting_data = sample_per_speaker(
        recordings,
        fitting_speakers,
        max_fit_frames_per_speaker,
        fit_frame_seed,
    )
    heldout_data = observations_for_speakers(recordings, heldout_speakers)
    metadata = json.loads(
        (tables_dir / "speaker_split.json").read_text(encoding="utf-8")
    )
    return fitting_data, heldout_data, metadata


def main() -> None:
    recordings = load_background_recordings(PROJECT_ROOT)
    create_split(recordings, PROJECT_ROOT / "results" / "task2" / "tables")
    print("Wrote the fixed 80/20 speaker split to results/task2/tables/")


if __name__ == "__main__":
    main()
