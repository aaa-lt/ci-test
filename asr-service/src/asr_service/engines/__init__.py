"""Recognition engines behind one interface, chosen by name (FR-26).

Engines import their heavy dependencies lazily, so the service starts without PyTorch
and simply reports unavailable engines in /health.
"""

from __future__ import annotations

import importlib
from functools import cache
from typing import Protocol

import numpy as np

from asr_service.types import Segment


class Engine(Protocol):
    name: str
    rate: int  # sample rate the engine expects
    max_window: float | None  # longest input in seconds; None means the whole channel at once
    word_level: bool

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        """Recognize mono float32 samples; times are relative to the first sample."""
        ...


# name -> (module, class, constructor kwargs)
REGISTRY: dict[str, tuple[str, str, dict[str, object]]] = {
    "gigaam-rnnt": ("asr_service.engines.gigaam", "GigaAM", {"model": "v3_rnnt"}),
    "gigaam-ctc": ("asr_service.engines.gigaam", "GigaAM", {"model": "v3_ctc"}),
    "tone": ("asr_service.engines.tone", "TOne", {"word_level": False}),
    "tone-greedy": ("asr_service.engines.tone", "TOne", {"word_level": True}),
    "whisper-turbo": ("asr_service.engines.whisper", "Whisper", {"model": "openai/whisper-large-v3-turbo"}),
    "yandex": ("asr_service.engines.yandex", "YandexSTT", {}),
}


@cache
def get_engine(name: str) -> Engine:
    if name not in REGISTRY:
        raise KeyError(f"unknown engine {name!r}; known: {', '.join(REGISTRY)}")
    module, cls, kwargs = REGISTRY[name]
    return getattr(importlib.import_module(module), cls)(name=name, **kwargs)


def available() -> dict[str, str]:
    """Engine name -> "ok" or the reason it cannot load, without loading weights."""
    out: dict[str, str] = {}
    for name, (module, cls, _) in REGISTRY.items():
        try:
            mod = importlib.import_module(module)
            check = getattr(getattr(mod, cls), "check", None)
            out[name] = check() if check else "ok"
        except ImportError as e:
            out[name] = f"missing dependency: {e.name}"
    return out
