"""T-One (streaming CTC, 8 kHz) with phrase or word timestamps.

The public pipeline returns phrases only: `TextPhrase(text, start_time, end_time)`.
Word timestamps are not part of the package, so `tone-greedy` derives them here from the
same per-frame log-probabilities: greedy CTC path, characters grouped into words at the
space token. `tone` keeps the package's beam search with its KenLM model and stays
phrase-level, because beam search output has no frame alignment.
"""

from __future__ import annotations

import numpy as np
from tone import StreamingCTCPipeline
from tone.decoder import LABELS, DecoderType
from tone.onnx_wrapper import StreamingCTCModel

from asr_service.ctc import ctc_words
from asr_service.types import Segment, Word

SPACE = LABELS.index(" ")


class TOne:
    rate = StreamingCTCModel.SAMPLE_RATE
    max_window = None  # streaming model, no length limit
    word_level: bool

    def __init__(self, name: str, word_level: bool) -> None:
        self.name = name
        self.word_level = word_level
        decoder = DecoderType.GREEDY if word_level else DecoderType.BEAM_SEARCH
        self.pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=decoder)

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        """Same chunk loop as `StreamingCTCPipeline.forward_offline`, keeping phrase log-probs."""
        p = self.pipeline
        audio = (np.clip(samples, -1, 1) * 32767).astype(np.int32)
        audio = np.pad(audio, (p.PADDING, p.PADDING))
        audio = np.pad(audio, (0, -len(audio) % p.CHUNK_SIZE))
        chunks = np.split(audio, len(audio) // p.CHUNK_SIZE)

        segments: list[Segment] = []
        model_state = splitter_state = None
        for i, chunk in enumerate(chunks):
            logprobs, model_state = p.model.forward(chunk[None, :, None], model_state)
            phrases, splitter_state = p.logprob_splitter.forward(
                logprobs[0], splitter_state, is_last=i == len(chunks) - 1
            )
            for phrase in phrases:
                lp = np.asarray(phrase.logprobs, dtype=np.float32)
                start = self._time(phrase.start_frame)
                # The splitter keeps SPEECH_EXPAND_SIZE frames of context before the phrase,
                # so row 0 of `lp` is that many frames before start_frame (off by at most
                # three frames for a phrase at the very start of the stream).
                lp_first = max(0, phrase.start_frame - p.logprob_splitter.SPEECH_EXPAND_SIZE)
                segments.append(
                    Segment(
                        text=p.decoder.forward(lp),
                        start=start,
                        end=max(start, self._time(phrase.end_frame)),
                        words=self._words(lp, lp_first) if self.word_level else None,
                    )
                )
        return [s for s in segments if s.text]

    def _time(self, frame: int) -> float:
        """Frame index to seconds, with the same bias and padding correction as the package."""
        m = StreamingCTCModel
        return max(
            0.0, round(frame * m.FRAME_SIZE - m.MEAN_TIME_BIAS - self.pipeline.PADDING / m.SAMPLE_RATE, 3)
        )

    def _words(self, logprobs: np.ndarray, first_frame: int) -> list[Word]:
        return ctc_words(
            logprobs.argmax(axis=-1).tolist(), LABELS, SPACE, lambda f: self._time(first_frame + f)
        )
