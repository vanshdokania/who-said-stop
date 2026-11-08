"""Audio loading and validation for the feature extraction pipeline."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

import torch
import torchaudio


EXPECTED_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioRecording:
    """A loaded mono recording and its basic timing information."""

    waveform: torch.Tensor
    sample_rate: int
    duration_seconds: float


def load_audio(audio_path: str | Path) -> AudioRecording:
    """Load a mono, 16 kHz audio file and return its waveform and metadata."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Audio path is not a file: {path}")

    try:
        ffmpeg_executable = shutil.which("ffmpeg") if os.name == "nt" else None
        if ffmpeg_executable is None:
            waveform, sample_rate = torchaudio.load(str(path))
        else:
            ffmpeg_directory = str(Path(ffmpeg_executable).parent)
            with os.add_dll_directory(ffmpeg_directory):
                waveform, sample_rate = torchaudio.load(str(path))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not decode audio file '{path}': {error}") from error

    if waveform.ndim != 2:
        raise ValueError(
            f"Expected a [channels, samples] waveform in '{path}', "
            f"got shape {tuple(waveform.shape)}"
        )
    if waveform.shape[0] != 1:
        raise ValueError(
            f"Expected mono audio in '{path}', got {waveform.shape[0]} channels"
        )
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Expected a {EXPECTED_SAMPLE_RATE} Hz sample rate in '{path}', "
            f"got {sample_rate} Hz"
        )

    duration_seconds = waveform.shape[1] / sample_rate
    return AudioRecording(
        waveform=waveform,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )
