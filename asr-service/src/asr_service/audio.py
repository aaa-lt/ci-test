"""Audio decoding through ffmpeg: any supported container in, float32 channels out."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np


def probe_channels(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return int(json.loads(out)["streams"][0]["channels"])


def decode(path: Path, rate: int) -> np.ndarray:
    """Decode to shape (channels, samples) float32 in [-1, 1], resampled to `rate`.

    Channels are kept separate: a stereo call yields one row per party (FR-13).
    """
    channels = probe_channels(path)
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples.reshape(-1, channels).T.copy()


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """Write mono float32 samples as 16-bit PCM; used for engines that only accept files."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
