"""Shared configuration for Task 1 feature extraction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters shared by every dataset split."""

    sample_rate: int = 16_000
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    silence_threshold_dbfs: float = -50.0
    low_energy_relative_threshold: float = 1e-4

    @property
    def frame_length_samples(self) -> int:
        return round(self.sample_rate * self.frame_length_ms / 1000)

    @property
    def hop_length_samples(self) -> int:
        return round(self.sample_rate * self.hop_length_ms / 1000)

    @property
    def n_fft(self) -> int:
        return self.frame_length_samples

    @property
    def num_frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def feature_dimension(self) -> int:
        return 2 * self.num_frequency_bins

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_length_ms <= 0 or self.hop_length_ms <= 0:
            raise ValueError("Frame and hop durations must be positive")
        if self.frame_length_samples <= 0 or self.hop_length_samples <= 0:
            raise ValueError("Frame and hop durations must round to at least one sample")
        if self.hop_length_samples > self.frame_length_samples:
            raise ValueError("hop_length_ms must not exceed frame_length_ms")
        if self.silence_threshold_dbfs >= 0:
            raise ValueError("silence_threshold_dbfs must be below 0 dBFS")
        if not 0 < self.low_energy_relative_threshold < 1:
            raise ValueError("low_energy_relative_threshold must be between 0 and 1")


DEFAULT_CONFIG = FeatureConfig()
