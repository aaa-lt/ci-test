"""GigaAM v3 through its native package; word timestamps come from the package itself."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gigaam
import numpy as np
import torch

from asr_service.audio import write_wav
from asr_service.types import Segment, Word


class GigaAM:
    rate = 16000
    # transcribe() rejects input over 25 s; stay below it so padding never trips the check.
    max_window = 24.0
    word_level = True

    def __init__(self, name: str, model: str) -> None:
        self.name = name
        # FP16 encoder weights only pay off on a GPU; on CPU they are slower or unsupported.
        self.model = gigaam.load_model(model, fp16_encoder=torch.cuda.is_available())

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.wav"
            write_wav(path, samples, self.rate)
            result = self.model.transcribe(str(path), word_timestamps=True)
        words = [Word(text=w.text, start=w.start, end=w.end) for w in result.words or []]
        if not words:
            return []
        return [Segment(text=result.text, start=words[0].start, end=words[-1].end, words=words)]
