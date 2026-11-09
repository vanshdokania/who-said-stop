"""Energy-based speech-frame detection; energy is never returned as a feature."""

import torch


def detect_speech_frames(
    frames: torch.Tensor,
    threshold_dbfs: float,
) -> torch.Tensor:
    """Mark frames whose RMS level is at or above the configured dBFS threshold."""
    if frames.ndim != 2:
        raise ValueError(f"Expected [num_frames, samples], got {frames.ndim} dimensions")
    if not frames.is_floating_point():
        raise TypeError("frames must have a floating-point dtype")
    if threshold_dbfs >= 0:
        raise ValueError("threshold_dbfs must be below 0 dBFS")
    if not torch.isfinite(frames).all():
        raise ValueError("frames contain NaN or infinite values")

    if frames.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=frames.device)

    rms = frames.square().mean(dim=1).sqrt()
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    return rms >= threshold
