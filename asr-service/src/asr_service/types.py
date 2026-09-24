"""Transcript types shared by engines, the stitcher and the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel


class Word(BaseModel):
    text: str
    start: float  # seconds from the start of the recording
    end: float


class Segment(BaseModel):
    """A stretch of speech from one channel.

    Word-level engines fill `words`; phrase-level engines (T-One with beam search) leave it
    empty, and consumers fall back to the segment start for seeking (NFR-7).
    """

    text: str
    start: float
    end: float
    words: list[Word] | None = None


class ChannelTranscript(BaseModel):
    channel: int
    role: str | None
    segments: list[Segment]
    windows: int  # number of engine calls made for this channel
    audio_seconds: float  # speech actually sent to the engine, after VAD


class Transcript(BaseModel):
    engine: str
    duration: float
    word_level: bool
    channels: list[ChannelTranscript]
    timings: dict[str, float]  # seconds: decode, vad, recognize, total
