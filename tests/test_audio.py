import struct
import tempfile
import unittest
import wave
from pathlib import Path

import torch

from src.features.audio import AudioRecording, load_audio


class AudioLoaderTests(unittest.TestCase):
    @staticmethod
    def write_pcm_wave(
        path: Path,
        channels: int = 1,
        sample_rate: int = 16_000,
        samples: tuple[int, ...] = (0, 16_384, -16_384, 32_767),
    ) -> None:
        with wave.open(str(path), "wb") as audio_file:
            audio_file.setnchannels(channels)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            audio_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    def test_load_returns_waveform_rate_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "recording.wav"
            self.write_pcm_wave(path)
            recording = load_audio(path)

        self.assertIsInstance(recording, AudioRecording)
        self.assertEqual(tuple(recording.waveform.shape), (1, 4))
        self.assertEqual(recording.waveform.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(
                recording.waveform,
                torch.tensor([[0.0, 0.5, -0.5, 32767 / 32768]], dtype=torch.float32),
            )
        )
        self.assertEqual(recording.sample_rate, 16_000)
        self.assertEqual(recording.duration_seconds, 4 / 16_000)

    def test_missing_path_has_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_audio(Path("missing-audio.wav"))

    def test_non_mono_audio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stereo.wav"
            self.write_pcm_wave(path, channels=2)
            with self.assertRaisesRegex(ValueError, "Expected mono audio"):
                load_audio(path)

    def test_wrong_sample_rate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "wrong-rate.wav"
            self.write_pcm_wave(path, sample_rate=8_000)
            with self.assertRaisesRegex(ValueError, "Expected a 16000 Hz"):
                load_audio(path)

    def test_unsupported_file_is_reported_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "unsupported.bin"
            path.write_bytes(b"not a wav file")
            with self.assertRaises(ValueError) as error:
                load_audio(path)
            self.assertIn(str(path), str(error.exception))

    def test_non_16_bit_pcm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "eight-bit.wav"
            with wave.open(str(path), "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(1)
                audio_file.setframerate(16_000)
                audio_file.writeframes(b"\x80\x80")
            with self.assertRaisesRegex(ValueError, "Expected 16-bit PCM"):
                load_audio(path)


if __name__ == "__main__":
    unittest.main()
