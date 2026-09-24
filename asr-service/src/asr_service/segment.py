"""Grouping a channel's words into utterances."""

from __future__ import annotations

from asr_service.types import Segment, Word


def words_to_segments(words: list[Word], max_pause: float = 0.8) -> list[Segment]:
    """Start a new segment whenever the pause between two words exceeds `max_pause` seconds.

    Each channel holds one party, so a long pause is the only boundary signal we have;
    the other party's turn shows up as silence on this channel.
    """
    segments: list[list[Word]] = []
    for w in words:
        if segments and w.start - segments[-1][-1].end <= max_pause:
            segments[-1].append(w)
        else:
            segments.append([w])
    return [
        Segment(text=" ".join(w.text for w in ws), start=ws[0].start, end=ws[-1].end, words=ws)
        for ws in segments
    ]
