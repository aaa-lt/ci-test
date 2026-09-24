"""Whole-recording recognition: decode, split channels, window, recognize, stitch."""

from __future__ import annotations

import time
from pathlib import Path

from asr_service import chunking
from asr_service.audio import decode
from asr_service.engines import get_engine
from asr_service.segment import words_to_segments
from asr_service.types import ChannelTranscript, Segment, Transcript


def transcribe_file(path: Path, engine_name: str, roles: list[str | None] | None = None) -> Transcript:
    """Recognize every channel of `path` separately; `roles[i]` labels channel i."""
    t0 = time.perf_counter()
    engine = get_engine(engine_name)
    load = time.perf_counter() - t0

    t1 = time.perf_counter()
    audio = decode(path, engine.rate)
    decode_s = time.perf_counter() - t1
    roles = roles or [None] * audio.shape[0]
    if len(roles) != audio.shape[0]:
        raise ValueError(f"{len(roles)} roles for {audio.shape[0]} channels")

    vad_s = recognize_s = 0.0
    channels: list[ChannelTranscript] = []
    for idx, samples in enumerate(audio):
        t2 = time.perf_counter()
        if engine.max_window is None:
            windows = [chunking.Window(0.0, len(samples) / engine.rate, 0.0, len(samples) / engine.rate)]
        else:
            energy = chunking.frame_energy_db(samples, engine.rate)
            regions = chunking.speech_regions(energy)
            windows = chunking.plan_windows(regions, energy, engine.max_window)
        vad_s += time.perf_counter() - t2

        t3 = time.perf_counter()
        per_window: list[list[Segment]] = []
        for w in windows:
            offset = w.start
            piece = samples[int(w.start * engine.rate) : int(w.end * engine.rate)]
            per_window.append([_shift(s, offset) for s in engine.transcribe(piece)])
        recognize_s += time.perf_counter() - t3

        if engine.word_level:
            words = chunking.stitch(
                windows, [[w for s in segs for w in s.words or []] for segs in per_window]
            )
            segments = words_to_segments(words)
        else:
            segments = [s for segs in per_window for s in segs]
        channels.append(
            ChannelTranscript(
                channel=idx,
                role=roles[idx],
                segments=segments,
                windows=len(windows),
                audio_seconds=round(sum(w.end - w.start for w in windows), 3),
            )
        )

    duration = audio.shape[1] / engine.rate
    return Transcript(
        engine=engine_name,
        duration=round(duration, 3),
        word_level=engine.word_level,
        channels=channels,
        timings={
            "load": round(load, 3),
            "decode": round(decode_s, 3),
            "vad": round(vad_s, 3),
            "recognize": round(recognize_s, 3),
            "total": round(time.perf_counter() - t0, 3),
            # NFR-13: recognition time over recording length, per channel pass
            "rtf": round(recognize_s / (duration * audio.shape[0]), 4) if duration else 0.0,
        },
    )


def _shift(segment: Segment, offset: float) -> Segment:
    words = [
        w.model_copy(update={"start": w.start + offset, "end": w.end + offset}) for w in segment.words or []
    ]
    return segment.model_copy(
        update={"start": segment.start + offset, "end": segment.end + offset, "words": words or segment.words}
    )
