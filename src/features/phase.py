"""STFT calculation and wrap-aware temporal phase-difference features."""

import torch


def calculate_stft(
    waveform: torch.Tensor,
    frame_length: int,
    hop_length: int,
) -> torch.Tensor:
    """Calculate a complex Hamming-window STFT with shape [frequency, time]."""
    if waveform.ndim != 1:
        raise ValueError(f"Expected a one-dimensional waveform, got {waveform.ndim} dimensions")
    if not waveform.is_floating_point():
        raise TypeError("waveform must have a floating-point dtype")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if waveform.numel() < frame_length:
        return torch.empty(
            (frame_length // 2 + 1, 0),
            dtype=torch.complex64 if waveform.dtype == torch.float32 else torch.complex128,
            device=waveform.device,
        )
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or infinite values")

    window = torch.hamming_window(
        frame_length,
        periodic=True,
        dtype=waveform.dtype,
        device=waveform.device,
    )
    return torch.stft(
        waveform,
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )


def temporal_phase_features(
    stft: torch.Tensor,
    relative_magnitude_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode adjacent-frame phase changes as cosine/sine pairs.

    Returns a [time-1, 2*frequency] feature tensor and a [time-1] mask
    identifying pairs with at least one reliable frequency bin.
    """
    if stft.ndim != 2 or not stft.is_complex():
        raise ValueError("stft must be a complex [frequency, time] tensor")
    if not 0 < relative_magnitude_threshold < 1:
        raise ValueError("relative_magnitude_threshold must be between 0 and 1")

    num_frequencies, num_frames = stft.shape
    if num_frames < 2:
        return (
            stft.real.new_empty((0, 2 * num_frequencies)),
            torch.zeros(0, dtype=torch.bool, device=stft.device),
        )
    if not torch.isfinite(stft.real).all() or not torch.isfinite(stft.imag).all():
        raise ValueError("stft contains NaN or infinite values")

    magnitude = stft.abs()
    frame_peak = magnitude.amax(dim=0, keepdim=True)
    reliable_bins = (magnitude > 0) & (
        magnitude >= frame_peak * relative_magnitude_threshold
    )
    reliable_pairs = reliable_bins[:, 1:] & reliable_bins[:, :-1]

    previous = stft[:, :-1]
    current = stft[:, 1:]
    cross_spectrum = current * previous.conj()
    cross_magnitude = cross_spectrum.abs()
    unit_cross_spectrum = cross_spectrum / cross_magnitude.clamp_min(
        torch.finfo(magnitude.dtype).tiny
    )

    cosine = torch.where(reliable_pairs, unit_cross_spectrum.real, 0.0)
    sine = torch.where(reliable_pairs, unit_cross_spectrum.imag, 0.0)
    features = torch.cat((cosine.transpose(0, 1), sine.transpose(0, 1)), dim=1)
    has_reliable_bin = reliable_pairs.any(dim=0)
    return features, has_reliable_bin
