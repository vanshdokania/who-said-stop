"""Extract and save Task 1 features for the competition dataset."""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_CONFIG
from src.features.audio import load_audio
from src.features.extractor import extract_features


SPLIT_MANIFESTS = {
    "background": ("background.csv",),
    "dev": ("dev_enrol.csv", "dev_trials.csv"),
    "eval": ("eval_enrol.csv", "eval_trials.csv"),
}
FEATURE_METHOD = "temporal_phase_difference_unit_circle"


def read_split_rows(
    data_dir: Path,
    manifest_names: tuple[str, ...],
) -> list[tuple[str, int, str, dict[str, str]]]:
    """Read manifest rows and return each audio-file reference with its context."""
    references: list[tuple[str, int, str, dict[str, str]]] = []
    for manifest_name in manifest_names:
        manifest_path = data_dir / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Required dataset manifest is missing: {manifest_path}")
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"Dataset manifest has no header: {manifest_path}")
            reference_column = (
                "file_id" if "file_id" in reader.fieldnames
                else "test_file" if "test_file" in reader.fieldnames
                else None
            )
            if reference_column is None:
                raise ValueError(
                    f"Manifest {manifest_path} must contain a file_id or test_file column"
                )
            for row_number, row in enumerate(reader, start=2):
                audio_id = (row.get(reference_column) or "").strip()
                if not audio_id:
                    raise ValueError(
                        f"Empty audio ID in {manifest_path} at row {row_number}"
                    )
                references.append((manifest_name, row_number, audio_id, row))
    return references


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Write a feature tensor atomically so an interrupted save is not mistaken for complete."""
    temporary_path = path.with_suffix(".pt.tmp")
    torch.save(tensor.cpu(), temporary_path)
    temporary_path.replace(path)


def extract_split(
    split: str,
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Extract unique recordings for a split and save a CSV index and settings."""
    references = read_split_rows(data_dir, SPLIT_MANIFESTS[split])
    unique_ids = dict.fromkeys(reference[2] for reference in references)
    audio_dir = data_dir / "audio" / "audio"
    resolved_paths = {
        audio_id: audio_dir / f"{audio_id}.wav"
        for audio_id in unique_ids
    }
    missing = [path for path in resolved_paths.values() if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} manifest audio file(s) are missing. Examples: {preview}"
        )

    split_output = output_dir / split
    feature_dir = split_output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    index_path = split_output / "index.csv"
    metadata_path = split_output / "metadata.json"
    expected_config = {
        **asdict(DEFAULT_CONFIG),
        "frame_length_samples": DEFAULT_CONFIG.frame_length_samples,
        "hop_length_samples": DEFAULT_CONFIG.hop_length_samples,
        "n_fft": DEFAULT_CONFIG.n_fft,
        "num_frequency_bins": DEFAULT_CONFIG.num_frequency_bins,
        "feature_dimension": DEFAULT_CONFIG.feature_dimension,
        "window": "periodic_hamming",
        "stft_center": False,
        "feature_method": FEATURE_METHOD,
        "silence_method": "frame_rms_dbfs",
        "normalization": "per_frame_l2",
    }
    can_reuse_features = False
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("config") != expected_config:
            raise ValueError(
                f"Existing features in {split_output} use different settings; "
                "choose a new output directory before extracting."
            )
        can_reuse_features = True

    print(f"{split}: extracting {len(unique_ids)} unique recordings")
    for index, (audio_id, audio_path) in enumerate(resolved_paths.items(), start=1):
        feature_path = feature_dir / f"{audio_id}.pt"
        if not can_reuse_features or not feature_path.is_file():
            recording = load_audio(audio_path)
            features = extract_features(
                recording.waveform,
                recording.sample_rate,
                DEFAULT_CONFIG,
            )
            save_tensor(feature_path, features)
        if index % 100 == 0 or index == len(unique_ids):
            print(f"{split}: {index}/{len(unique_ids)} recordings ready")

    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("manifest", "row_number", "audio_id", "feature_path", "reference_json"),
        )
        writer.writeheader()
        for manifest, row_number, audio_id, row in references:
            writer.writerow(
                {
                    "manifest": manifest,
                    "row_number": row_number,
                    "audio_id": audio_id,
                    "feature_path": str(Path("features") / f"{audio_id}.pt"),
                    "reference_json": json.dumps(row, sort_keys=True),
                }
            )

    metadata: dict[str, Any] = {
        "split": split,
        "feature_count": len(unique_ids),
        "reference_count": len(references),
        "manifests": list(SPLIT_MANIFESTS[split]),
        "feature_shape": ["retained_frames", DEFAULT_CONFIG.feature_dimension],
        "config": expected_config,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{split}: wrote features, index, and metadata to {split_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=(*SPLIT_MANIFESTS, "all"),
        default="all",
        help="Dataset split to process (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing the original CSVs and audio folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "features_output",
        help="Directory for generated features and metadata",
    )
    args = parser.parse_args()

    selected_splits = SPLIT_MANIFESTS if args.split == "all" else (args.split,)
    for split in selected_splits:
        extract_split(split, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
