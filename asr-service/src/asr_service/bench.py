"""WER of the local engines on real telephone speech: the Open STT phone-call validation set.

Runs inside the Space so the audio never crosses the user's connection. The HF mirror
`Sh1man/silero_open_stt` holds exactly the manually annotated `*_val` subsets of Open STT
(12 950 + 7 850 + 7 311 = 28 111 utterances); `asr_calls_v2` is the phone-call one.
License CC BY-NC 4.0: used as an external benchmark, not as the project's own data set.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import jiwer
from datasets import Audio, load_dataset

from asr_service.textnorm import normalize
from asr_service.transcribe import transcribe_file

DATASET = "Sh1man/silero_open_stt"
CONFIG = "asr_calls_v2"
SPLIT = "validate"


def load_samples(n: int, min_duration: float = 1.0) -> list[dict[str, Any]]:
    """First `n` utterances of at least `min_duration` seconds, in the archive's order."""
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True).cast_column("wav", Audio(decode=False))
    out: list[dict[str, Any]] = []
    for row in ds:
        meta = row["json"]
        if meta["duration"] >= min_duration and meta["text"].strip():
            out.append(
                {
                    "id": meta["id"],
                    "duration": meta["duration"],
                    "text": meta["text"],
                    "wav": row["wav"]["bytes"],
                }
            )
            if len(out) == n:
                break
    return out


def run(engines: list[str], n: int = 150) -> dict[str, Any]:
    samples = load_samples(n)
    report: dict[str, Any] = {
        "dataset": f"{DATASET}/{CONFIG}/{SPLIT}",
        "samples": len(samples),
        "audio_seconds": round(sum(s["duration"] for s in samples), 2),
        "engines": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for s in samples:
            p = Path(tmp) / f"{s['id']}.wav"
            p.write_bytes(s["wav"])
            paths.append(p)
        for engine in engines:
            refs, hyps, items = [], [], []
            recognize = 0.0
            t0 = time.perf_counter()
            for s, p in zip(samples, paths, strict=True):
                tr = transcribe_file(p, engine)
                recognize += tr.timings["recognize"]
                hyp = " ".join(seg.text for ch in tr.channels for seg in ch.segments)
                ref_n, hyp_n = normalize(s["text"]), normalize(hyp)
                refs.append(ref_n)
                hyps.append(hyp_n)
                items.append({"id": s["id"], "ref": ref_n, "hyp": hyp_n})
            measures = jiwer.process_words(refs, hyps)
            report["engines"][engine] = {
                "wer": round(measures.wer, 4),
                "substitutions": measures.substitutions,
                "deletions": measures.deletions,
                "insertions": measures.insertions,
                "reference_words": sum(len(r.split()) for r in refs),
                "rtf": round(recognize / report["audio_seconds"], 4),
                "wall_seconds": round(time.perf_counter() - t0, 1),
                "items": items,
            }
    return report
