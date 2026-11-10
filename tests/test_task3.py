import csv
import tempfile
import unittest
from pathlib import Path

from experiments.task3.prepare_data import load_task2_split
from src.utils.data_loader import BackgroundRecording
import torch


class Task3DataPreparationTests(unittest.TestCase):
    def test_reads_fixed_task2_split_and_validates_speaker_mapping(self) -> None:
        recordings = [
            BackgroundRecording("file_a", "speaker_a", torch.ones((2, 3))),
            BackgroundRecording("file_b", "speaker_b", torch.zeros((2, 3))),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_dir = root / "results" / "task2" / "tables"
            split_dir.mkdir(parents=True)
            with (split_dir / "speaker_split.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("file_id", "speaker_id", "subset"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "file_id": "file_a",
                        "speaker_id": "speaker_a",
                        "subset": "fitting",
                    }
                )
                writer.writerow(
                    {
                        "file_id": "file_b",
                        "speaker_id": "speaker_b",
                        "subset": "heldout",
                    }
                )
            fitting, heldout = load_task2_split(root, recordings)

        self.assertEqual(fitting, {"speaker_a"})
        self.assertEqual(heldout, {"speaker_b"})
        self.assertFalse(fitting & heldout)

    def test_rejects_split_with_speaker_mapping_mismatch(self) -> None:
        recordings = [
            BackgroundRecording("file_a", "speaker_a", torch.ones((2, 3))),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_dir = root / "results" / "task2" / "tables"
            split_dir.mkdir(parents=True)
            with (split_dir / "speaker_split.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("file_id", "speaker_id", "subset"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "file_id": "file_a",
                        "speaker_id": "wrong_speaker",
                        "subset": "fitting",
                    }
                )
            with self.assertRaisesRegex(ValueError, "speaker mapping disagrees"):
                load_task2_split(root, recordings)


if __name__ == "__main__":
    unittest.main()
