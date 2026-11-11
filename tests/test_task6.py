import tempfile
import unittest
from pathlib import Path

import torch

from experiments.task6.plots import plot_enrolment_size
from experiments.task6.run_all import (
    _load_selected_enrolment_features,
    _read_dev_enrolments,
)


class Task6Tests(unittest.TestCase):
    def test_selects_first_n_recordings_by_enrol_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_root = root / "features_output" / "dev" / "features"
            feature_root.mkdir(parents=True)
            features = {
                "late": torch.full((2, 402), 3.0),
                "first": torch.full((1, 402), 1.0),
                "middle": torch.full((3, 402), 2.0),
            }
            for file_id, tensor in features.items():
                torch.save(tensor, feature_root / f"{file_id}.pt")

            references = {"speaker": [(2, "late"), (0, "first"), (1, "middle")]}
            # The loader uses the project-level path; temporarily place the files
            # in its expected root through a direct path setup below.
            from experiments.task6 import run_all

            original_root = run_all.PROJECT_ROOT
            run_all.PROJECT_ROOT = root
            try:
                pooled, file_ids = _load_selected_enrolment_features(references, 2)
            finally:
                run_all.PROJECT_ROOT = original_root

            self.assertEqual(file_ids["speaker"], ["first", "middle"])
            self.assertEqual(tuple(pooled["speaker"].shape), (4, 402))
            self.assertTrue(torch.equal(pooled["speaker"][:1], features["first"].to(torch.float64)))
            self.assertTrue(torch.equal(pooled["speaker"][1:], features["middle"].to(torch.float64)))

    def test_enrolment_size_plot_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "enrolment.svg"
            plot_enrolment_size(
                [
                    {"enrolment_utterances": 1, "eer": 0.3},
                    {"enrolment_utterances": 2, "eer": 0.2},
                    {"enrolment_utterances": 3, "eer": 0.1},
                ],
                output,
            )
            self.assertIn("<svg", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
