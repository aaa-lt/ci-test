"""Cutting a channel into engine-sized windows and stitching the words back (FR-24).

The plan has two levels:

1. An energy detector finds speech regions; silence between them is never sent to the
   engine. Regions are packed into windows no longer than the engine limit, and cuts
   between windows fall into silence, so no word can straddle them.
2. A single region longer than the limit (continuous speech) is split at the quietest
   point near the limit, and the neighbouring windows overlap by `overlap` seconds on
   each side of the cut. Every word is owned by the window whose core contains its
   midpoint, so a word at the cut is taken exactly once.

All functions work in seconds on plain lists, so they can be tested without audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asr_service.types import Word

FRAME = 0.02  # energy frame, seconds


@dataclass(frozen=True)
class Window:
    start: float  # audio actually sent to the engine
    end: float
    core_start: float  # words with midpoints in [core_start, core_end) belong to this window
    core_end: float


def frame_energy_db(samples: np.ndarray, rate: int) -> np.ndarray:
    n = int(FRAME * rate)
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def speech_regions(
    energy_db: np.ndarray,
    margin_db: float = 15.0,
    min_gap: float = 0.3,
    min_speech: float = 0.15,
    pad: float = 0.2,
) -> list[tuple[float, float]]:
    """Speech regions as (start, end) seconds from per-frame energy.

    A frame is speech when it is `margin_db` above the noise floor (10th percentile).
    Regions closer than `min_gap` are merged, shorter than `min_speech` dropped, and
    each is widened by `pad` so that soft word onsets and endings are kept.
    """
    if energy_db.size == 0:
        return []
    floor = float(np.percentile(energy_db, 10))
    active = energy_db > floor + margin_db
    regions: list[list[float]] = []
    for i in np.flatnonzero(active):
        t = i * FRAME
        if regions and t - regions[-1][1] <= min_gap:
            regions[-1][1] = t + FRAME
        else:
            regions.append([t, t + FRAME])
    total = energy_db.size * FRAME
    return [(max(0.0, s - pad), min(total, e + pad)) for s, e in regions if e - s >= min_speech]


def plan_windows(
    regions: list[tuple[float, float]],
    energy_db: np.ndarray,
    max_len: float,
    overlap: float = 1.0,
    search: float = 6.0,
) -> list[Window]:
    """Pack speech regions into windows of at most `max_len` seconds."""
    windows: list[Window] = []
    group: list[tuple[float, float]] = []

    def flush() -> None:
        if group:
            windows.append(Window(group[0][0], group[-1][1], group[0][0], group[-1][1]))
            group.clear()

    for start, end in regions:
        if end - start > max_len:
            flush()
            windows.extend(split_region(start, end, energy_db, max_len, overlap, search))
        elif group and end - group[0][0] > max_len:
            flush()
            group.append((start, end))
        else:
            group.append((start, end))
    flush()

    # Cores of silence-separated windows meet halfway through the gap, so ownership is total.
    for i in range(1, len(windows)):
        prev, cur = windows[i - 1], windows[i]
        if prev.core_end <= cur.core_start:
            mid = (prev.core_end + cur.core_start) / 2
            windows[i - 1] = Window(prev.start, prev.end, prev.core_start, mid)
            windows[i] = Window(cur.start, cur.end, mid, cur.core_end)
    return windows


def split_region(
    start: float, end: float, energy_db: np.ndarray, max_len: float, overlap: float, search: float
) -> list[Window]:
    """Split continuous speech at the quietest frames; windows overlap around each cut."""
    step = max_len - 2 * overlap
    if step <= 0:
        raise ValueError("max_len must exceed twice the overlap")
    cuts = [start]
    while end - cuts[-1] > max_len - overlap:
        hi = cuts[-1] + step
        lo = max(cuts[-1] + step / 2, hi - search)
        a, b = int(lo / FRAME), max(int(lo / FRAME) + 1, int(hi / FRAME))
        cuts.append((a + int(np.argmin(energy_db[a:b]))) * FRAME if b <= energy_db.size else hi)
    cuts.append(end)
    return [
        Window(
            start=max(start, cuts[i] - overlap),
            end=min(end, cuts[i + 1] + overlap),
            core_start=cuts[i],
            core_end=cuts[i + 1],
        )
        for i in range(len(cuts) - 1)
    ]


def stitch(windows: list[Window], words_per_window: list[list[Word]]) -> list[Word]:
    """Merge per-window words (already in absolute time) into one sequence.

    A word belongs to the window whose core holds its midpoint. If the two windows around
    a cut still both claim the same word with slightly different timing, the repeat is
    dropped: same text and more than half of the shorter word overlapping.
    """
    merged: list[Word] = []
    last = len(windows) - 1
    for i, (win, words) in enumerate(zip(windows, words_per_window, strict=True)):
        lo = -np.inf if i == 0 else win.core_start
        hi = np.inf if i == last else win.core_end
        for w in words:
            if not lo <= (w.start + w.end) / 2 < hi:
                continue
            if merged and _same_word(merged[-1], w):
                continue
            merged.append(w)
    merged.sort(key=lambda w: w.start)
    return merged


def _same_word(a: Word, b: Word) -> bool:
    if a.text.lower() != b.text.lower():
        return False
    inter = min(a.end, b.end) - max(a.start, b.start)
    shorter = min(a.end - a.start, b.end - b.start)
    return shorter > 0 and inter > shorter / 2
