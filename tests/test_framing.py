import unittest

import torch

from src.features.framing import frame_waveform


class FramingTests(unittest.TestCase):
    def test_overlapping_frame_count_and_contents(self) -> None:
        waveform = torch.arange(10, dtype=torch.float32)
        frames = frame_waveform(waveform, frame_length=4, hop_length=3)

        self.assertEqual(tuple(frames.shape), (3, 4))
        self.assertTrue(torch.equal(frames[0], torch.tensor([0.0, 1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(frames[1], torch.tensor([3.0, 4.0, 5.0, 6.0])))
        self.assertTrue(torch.equal(frames[2], torch.tensor([6.0, 7.0, 8.0, 9.0])))

    def test_short_waveform_has_no_padded_frames(self) -> None:
        frames = frame_waveform(torch.ones(3), frame_length=4, hop_length=2)
        self.assertEqual(tuple(frames.shape), (0, 4))

    def test_invalid_waveform_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            frame_waveform(torch.ones(2, 4), frame_length=4, hop_length=2)


if __name__ == "__main__":
    unittest.main()
