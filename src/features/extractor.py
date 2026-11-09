"""End-to-end phase-based feature extraction for one recording."""

import torch

from config import DEFAULT_CONFIG, FeatureConfig
from src.features.framing import frame_waveform
from src.features.normalization import normalize_feature_vectors
from src.features.phase import calculate_stft, temporal_phase_features
from src.features.silence import detect_speech_frames


def extract_features(
    waveform: torch.Tensor,
    sample_rate: int,
    config: FeatureConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Return a [retained_frames, feature_dimension] tensor for mono audio."""
    if waveform.ndim == 2:
        if waveform.shape[0] != 1:
            raise ValueError(f"Expected mono audio, got {waveform.shape[0]} channels")
        waveform = waveform.squeeze(0)
    elif waveform.ndim != 1:
        raise ValueError("waveform must have shape [samples] or [1, samples]")
    if not waveform.is_floating_point():
        raise TypeError("waveform must have a floating-point dtype")
    if sample_rate != config.sample_rate:
        raise ValueError(
            f"Expected a {config.sample_rate} Hz sample rate, got {sample_rate} Hz"
        )
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or infinite values")

    frames = frame_waveform(
        waveform,
        config.frame_length_samples,
        config.hop_length_samples,
    )
    empty = waveform.new_empty((0, config.feature_dimension))
    if frames.shape[0] < 2:
        return empty

    speech_frames = detect_speech_frames(frames, config.silence_threshold_dbfs)
    if not speech_frames.any():
        return empty

    stft = calculate_stft(
        waveform,
        config.frame_length_samples,
        config.hop_length_samples,
    )
    phase_features, has_reliable_bin = temporal_phase_features(
        stft,
        config.low_energy_relative_threshold,
    )
    voiced_pairs = speech_frames[:-1] & speech_frames[1:]
    retained_pairs = voiced_pairs & has_reliable_bin
    return normalize_feature_vectors(phase_features[retained_pairs])
