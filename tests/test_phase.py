import math
import unittest

import torch

from config import DEFAULT_CONFIG
from src.features.phase import calculate_stft, temporal_phase_features


class PhaseFeatureTests(unittest.TestCase):
    def test_temporal_difference_handles_phase_wrap(self) -> None:
        phases = torch.tensor([math.pi - 0.1, -math.pi + 0.1])
        stft = torch.polar(torch.ones(1, 2), phases).to(torch.complex64)

        features, has_reliable_bin = temporal_phase_features(
            stft,
            relative_magnitude_threshold=1e-4,
        )

        self.assertTrue(has_reliable_bin.item())
        self.assertAlmostEqual(features[0, 0].item(), math.cos(0.2), places=5)
        self.assertAlmostEqual(features[0, 1].item(), math.sin(0.2), places=5)

    def test_low_energy_bins_are_zeroed(self) -> None:
        stft = torch.tensor(
            [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1e-6 + 0.0j]],
            dtype=torch.complex64,
        )
        features, has_reliable_bin = temporal_phase_features(
            stft,
            relative_magnitude_threshold=1e-3,
        )

        self.assertTrue(has_reliable_bin.item())
        self.assertEqual(features[0, 0].item(), 1.0)
        self.assertEqual(features[0, 1].item(), 0.0)
        self.assertEqual(features[0, 2].item(), 0.0)
        self.assertEqual(features[0, 3].item(), 0.0)

    def test_stft_shape_matches_framing_configuration(self) -> None:
        waveform = torch.randn(4_000)
        stft = calculate_stft(
            waveform,
            DEFAULT_CONFIG.frame_length_samples,
            DEFAULT_CONFIG.hop_length_samples,
        )
        expected_frames = 1 + (
            waveform.numel() - DEFAULT_CONFIG.frame_length_samples
        ) // DEFAULT_CONFIG.hop_length_samples
        self.assertEqual(tuple(stft.shape), (DEFAULT_CONFIG.num_frequency_bins, expected_frames))
        self.assertTrue(stft.is_complex())

    def test_short_waveform_returns_empty_complex_stft(self) -> None:
        stft = calculate_stft(torch.ones(100), frame_length=400, hop_length=160)
        self.assertEqual(tuple(stft.shape), (201, 0))
        self.assertTrue(stft.is_complex())


if __name__ == "__main__":
    unittest.main()
