"""PCM WAV loading and validation for the feature extraction pipeline."""

import wave
from dataclasses import dataclass
from pathlib import Path

import torch


EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class AudioRecording:
    """A loaded mono recording and its basic timing information."""

    waveform: torch.Tensor
    sample_rate: int
    duration_seconds: float


def load_audio(audio_path: str | Path) -> AudioRecording:
    """Load mono, 16 kHz, 16-bit PCM WAV audio and return its waveform and metadata.

    Python's standard-library WAV reader avoids the blocked TorchCodec native
    extension. Samples are converted directly to normalized float32 Torch
    tensors; no NumPy or third-party decoder is used.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Audio path is not a file: {path}")

    try:
        with wave.open(str(path), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_rate = audio_file.getframerate()
            sample_width = audio_file.getsampwidth()
            frame_count = audio_file.getnframes()
            compression = audio_file.getcomptype()
            if compression != "NONE":
                raise ValueError(
                    f"Unsupported WAV compression in '{path}': {compression}"
                )
            if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
                raise ValueError(
                    f"Expected 16-bit PCM audio in '{path}', "
                    f"got {sample_width * 8}-bit samples"
                )
            raw_samples = audio_file.readframes(frame_count)
    except (wave.Error, EOFError, OSError) as error:
        raise ValueError(f"Could not decode audio file '{path}': {error}") from error

    expected_bytes = frame_count * channels * sample_width
    if len(raw_samples) != expected_bytes:
        raise ValueError(
            f"Truncated audio data in '{path}': expected {expected_bytes} bytes, "
            f"read {len(raw_samples)}"
        )
    if frame_count == 0:
        waveform = torch.empty((channels, 0), dtype=torch.float32)
    else:
        integer_samples = torch.frombuffer(
            bytearray(raw_samples),
            dtype=torch.int16,
        )
        waveform = (
            integer_samples.to(torch.float32)
            .div_(32768.0)
            .reshape(frame_count, channels)
            .transpose(0, 1)
            .contiguous()
        )

    if waveform.shape[0] != 1:
        raise ValueError(f"Expected mono audio in '{path}', got {waveform.shape[0]} channels")
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
