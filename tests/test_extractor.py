import unittest

import torch

from config import DEFAULT_CONFIG, FeatureConfig
from src.features.extractor import extract_features


class FeatureExtractorTests(unittest.TestCase):
    def test_output_is_finite_normalized_and_has_expected_dimension(self) -> None:
        sample_rate = DEFAULT_CONFIG.sample_rate
        time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        waveform = 0.2 * torch.sin(2 * torch.pi * 440 * time)

        features = extract_features(waveform, sample_rate)

        self.assertEqual(features.shape[1], DEFAULT_CONFIG.feature_dimension)
        self.assertGreater(features.shape[0], 0)
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(features, dim=1), torch.ones(features.shape[0])))

    def test_silent_audio_returns_empty_tensor(self) -> None:
        features = extract_features(torch.zeros(8_000), DEFAULT_CONFIG.sample_rate)
        self.assertEqual(tuple(features.shape), (0, DEFAULT_CONFIG.feature_dimension))

    def test_short_audio_returns_empty_tensor(self) -> None:
        features = extract_features(torch.ones(100), DEFAULT_CONFIG.sample_rate)
        self.assertEqual(tuple(features.shape), (0, DEFAULT_CONFIG.feature_dimension))

    def test_sample_rate_and_channels_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            extract_features(torch.ones(8_000), sample_rate=8_000)
        with self.assertRaises(ValueError):
            extract_features(torch.ones(2, 8_000), sample_rate=DEFAULT_CONFIG.sample_rate)

    def test_configuration_calculates_feature_dimension(self) -> None:
        config = FeatureConfig()
        self.assertEqual(config.frame_length_samples, 400)
        self.assertEqual(config.hop_length_samples, 160)
        self.assertEqual(config.num_frequency_bins, 201)
        self.assertEqual(config.feature_dimension, 402)


if __name__ == "__main__":
    unittest.main()
