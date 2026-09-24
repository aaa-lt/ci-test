"""Word timestamps from a greedy CTC path, for engines that only return phrases."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from asr_service.types import Word


def ctc_words(
    path: Sequence[int],
    labels: str,
    space: int,
    frame_time: Callable[[int], float],
) -> list[Word]:
    """Group a per-frame argmax path into words.

    `path[f]` is the most probable token at frame f; indices below `space` are letters of
    `labels`, `space` is the word separator and anything above it is the CTC blank.
    Repeated tokens collapse into one character unless a blank separates them. A word
    spans from the first frame of its first character to the frame after its last one.
    """
    words: list[Word] = []
    chars: list[str] = []
    first = last = 0
    prev = -1

    def commit() -> None:
        if chars:
            words.append(Word(text="".join(chars), start=frame_time(first), end=frame_time(last + 1)))
            chars.clear()

    for f, tok in enumerate(path):
        if tok == space:
            commit()
        elif tok < space:
            if tok != prev:
                if not chars:
                    first = f
                chars.append(labels[tok])
            last = f
        prev = tok
    commit()
    return words
