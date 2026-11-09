"""Short-time framing for mono waveforms."""

import torch


def frame_waveform(
    waveform: torch.Tensor,
    frame_length: int,
    hop_length: int,
) -> torch.Tensor:
    """Return overlapping frames with shape [num_frames, frame_length]."""
    if waveform.ndim != 1:
        raise ValueError(f"Expected a one-dimensional waveform, got {waveform.ndim} dimensions")
    if not waveform.is_floating_point():
        raise TypeError("waveform must have a floating-point dtype")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")

    if waveform.numel() < frame_length:
        return waveform.new_empty((0, frame_length))
    return waveform.unfold(0, frame_length, hop_length)
