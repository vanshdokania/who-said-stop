import unittest

import torch

from src.features.silence import detect_speech_frames


class SilenceDetectionTests(unittest.TestCase):
    def test_rms_threshold_marks_quiet_and_loud_frames(self) -> None:
        frames = torch.stack(
            (
                torch.zeros(400),
                torch.full((400,), 0.001),
                torch.full((400,), 0.1),
            )
        )
        mask = detect_speech_frames(frames, threshold_dbfs=-50.0)
        self.assertTrue(torch.equal(mask, torch.tensor([False, False, True])))

    def test_empty_frames_return_empty_mask(self) -> None:
        mask = detect_speech_frames(torch.empty(0, 400), threshold_dbfs=-50.0)
        self.assertEqual(tuple(mask.shape), (0,))
        self.assertEqual(mask.dtype, torch.bool)

    def test_non_finite_samples_are_rejected(self) -> None:
        frames = torch.zeros(1, 400)
        frames[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            detect_speech_frames(frames, threshold_dbfs=-50.0)


if __name__ == "__main__":
    unittest.main()
