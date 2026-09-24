"""Yandex SpeechKit v3 asynchronous recognition, the cloud reference column.

Violates FR-22 by design (audio leaves the machine), so it is only used on synthetic
calls to compare against the local engines. Reads YANDEX_API_KEY and YANDEX_FOLDER_ID.
"""

from __future__ import annotations

import base64
import json
import os
import time

import httpx
import numpy as np

from asr_service.types import Segment, Word

RECOGNIZE = "https://stt.api.cloud.yandex.net/stt/v3/recognizeFileAsync"
RESULT = "https://stt.api.cloud.yandex.net/stt/v3/getRecognition"


class YandexSTT:
    rate = 8000
    max_window = None  # async API accepts the whole channel
    word_level = True

    @staticmethod
    def check() -> str:
        missing = [k for k in ("YANDEX_API_KEY", "YANDEX_FOLDER_ID") if not os.environ.get(k)]
        return f"missing env: {', '.join(missing)}" if missing else "ok"

    def __init__(self, name: str, poll: float = 2.0, timeout: float = 600.0) -> None:
        self.name = name
        self.poll = poll
        self.timeout = timeout
        self.client = httpx.Client(
            headers={
                "Authorization": f"Api-Key {os.environ['YANDEX_API_KEY']}",
                "x-folder-id": os.environ["YANDEX_FOLDER_ID"],
            },
            timeout=60,
        )

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        body = {
            "content": base64.b64encode(pcm).decode(),
            "recognitionModel": {
                "model": "general",
                "audioFormat": {
                    "rawAudio": {
                        "audioEncoding": "LINEAR16_PCM",
                        "sampleRateHertz": self.rate,
                        "audioChannelCount": 1,
                    }
                },
                # Keep numbers as words, like the other engines and the reference transcripts.
                "textNormalization": {"textNormalization": "TEXT_NORMALIZATION_DISABLED"},
                "languageRestriction": {"restrictionType": "WHITELIST", "languageCode": ["ru-RU"]},
            },
        }
        resp = self.client.post(RECOGNIZE, json=body)
        resp.raise_for_status()
        operation = resp.json()["id"]

        deadline = time.monotonic() + self.timeout
        while True:
            time.sleep(self.poll)
            result = self.client.get(RESULT, params={"operationId": operation})
            if result.status_code == 200 and result.content:
                return parse_results(result.text)
            if result.status_code not in (400, 404) or time.monotonic() > deadline:
                raise RuntimeError(
                    f"recognition {operation} failed: {result.status_code} {result.text[:300]}"
                )


def parse_results(body: str) -> list[Segment]:
    """Collect `final` events from the newline-delimited StreamingResponse stream."""
    segments: list[Segment] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event = event.get("result", event)
        final = event.get("final")
        if not final:
            continue
        for alt in final.get("alternatives", [])[:1]:
            words = [
                Word(text=w["text"], start=int(w["startTimeMs"]) / 1000, end=int(w["endTimeMs"]) / 1000)
                for w in alt.get("words", [])
            ]
            if words:
                segments.append(
                    Segment(text=alt.get("text", ""), start=words[0].start, end=words[-1].end, words=words)
                )
    return segments
