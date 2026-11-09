import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.extract_features import extract_split
from src.features.audio import AudioRecording


class DatasetExtractionTests(unittest.TestCase):
    def test_split_deduplicates_audio_and_preserves_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            audio_dir = data_dir / "audio" / "audio"
            audio_dir.mkdir(parents=True)

            with (data_dir / "dev_enrol.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("file_id", "speaker_id", "enrol_index"))
                writer.writeheader()
                writer.writerow({"file_id": "u_enrol", "speaker_id": "speaker_1", "enrol_index": "0"})
            with (data_dir / "dev_trials.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("trial_id", "claimed_id", "test_file", "label"))
                writer.writeheader()
                writer.writerow({"trial_id": "trial_1", "claimed_id": "speaker_1", "test_file": "u_test", "label": "1"})
                writer.writerow({"trial_id": "trial_2", "claimed_id": "speaker_2", "test_file": "u_test", "label": "0"})

            for audio_id in ("u_enrol", "u_test"):
                (audio_dir / f"{audio_id}.wav").touch()

            waveform = torch.sin(
                2 * torch.pi * 440
                * torch.arange(16_000, dtype=torch.float32)
                / 16_000
            ).unsqueeze(0)
            fake_recording = AudioRecording(
                waveform=waveform,
                sample_rate=16_000,
                duration_seconds=1.0,
            )
            output_dir = root / "features_output"
            with patch("scripts.extract_features.load_audio", return_value=fake_recording) as load:
                extract_split("dev", data_dir, output_dir)

            self.assertEqual(load.call_count, 2)
            split_output = output_dir / "dev"
            saved_features = list((split_output / "features").glob("*.pt"))
            self.assertEqual(len(saved_features), 2)
            for feature_path in saved_features:
                features = torch.load(feature_path, weights_only=True)
                self.assertEqual(features.shape[1], 402)
                self.assertTrue(torch.isfinite(features).all())

            with (split_output / "index.csv").open(newline="", encoding="utf-8") as file:
                index_rows = list(csv.DictReader(file))
            self.assertEqual(len(index_rows), 3)
            self.assertEqual(
                sum(row["audio_id"] == "u_test" for row in index_rows),
                2,
            )
            metadata = json.loads((split_output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["feature_count"], 2)
            self.assertEqual(metadata["reference_count"], 3)
            self.assertEqual(metadata["config"]["feature_dimension"], 402)


if __name__ == "__main__":
    unittest.main()
