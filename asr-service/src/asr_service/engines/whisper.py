"""Whisper through transformers on CPU.

On the Mac this engine is meant to run through mlx-whisper instead (NFR-28); MLX has no
Linux build, so the Space uses the transformers implementation of the same weights.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import pipeline

from asr_service.types import Segment, Word


class Whisper:
    rate = 16000
    max_window = 28.0  # the model sees 30 s at most; chunking is ours, not the pipeline's
    word_level = True

    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.pipe = pipeline("automatic-speech-recognition", model=model, device="cpu", dtype=torch.float32)

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        out = self.pipe(
            {"raw": samples, "sampling_rate": self.rate},
            return_timestamps="word",
            generate_kwargs={"language": "russian", "task": "transcribe"},
        )
        words = []
        for chunk in out.get("chunks", []):
            text = chunk["text"].strip()
            start, end = chunk["timestamp"]
            if text:
                words.append(Word(text=text, start=start, end=end if end is not None else start))
        if not words:
            return []
        return [Segment(text=out["text"].strip(), start=words[0].start, end=words[-1].end, words=words)]
