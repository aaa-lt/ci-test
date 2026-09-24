"""transcribe_file end to end with a fake engine: real ffmpeg decoding, VAD, windows, stitching."""

import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from asr_service import app as app_module
from asr_service import transcribe as transcribe_module
from asr_service.types import Segment, Word

RATE = 16000


class FakeEngine:
    """Reports one word per burst of tone it hears, placed where the burst is."""

    rate = RATE
    max_window = 5.0
    word_level = True

    def __init__(self) -> None:
        self.windows: list[float] = []

    def transcribe(self, samples: np.ndarray) -> list[Segment]:
        self.windows.append(len(samples) / RATE)
        frame = RATE // 100  # 10 ms envelope, so zero crossings of the tone do not split a word
        n = len(samples) // frame
        loud = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1)) > 0.1
        words, start = [], None
        for i, on in enumerate(np.append(loud, False)):
            if on and start is None:
                start = i
            elif not on and start is not None:
                if i - start > 10:
                    words.append(Word(text=f"w{len(words)}", start=start * 0.01, end=i * 0.01))
                start = None
        return [Segment(text=" ".join(w.text for w in words), start=0, end=0, words=words)] if words else []


def stereo_call(
    path: Path, left: list[tuple[float, float]], right: list[tuple[float, float]], total: float
) -> None:
    t = np.arange(int(total * RATE)) / RATE
    data = np.zeros((len(t), 2), dtype=np.float32)
    for ch, bursts in enumerate((left, right)):
        for s, e in bursts:
            m = (t >= s) & (t < e)
            data[m, ch] = 0.5 * np.sin(2 * np.pi * 440 * t[m])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((data * 32767).astype("<i2").tobytes())


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setattr(transcribe_module, "get_engine", lambda name: fake)
    return fake


def test_channels_are_recognized_separately_with_speech_only_windows(tmp_path: Path, engine: FakeEngine):
    wav = tmp_path / "call.wav"
    # operator speaks in three bursts spread over 12 s; the client once
    stereo_call(wav, left=[(1.0, 1.6), (3.0, 3.5), (9.0, 9.8)], right=[(5.0, 5.7)], total=12.0)
    tr = transcribe_module.transcribe_file(wav, "fake", ["operator", "client"])

    op, cl = tr.channels
    assert (op.role, cl.role) == ("operator", "client")
    assert [len(s.words or []) for s in op.segments] == [1, 1, 1]  # pauses > 0.8 s split segments
    starts = [s.start for s in op.segments]
    assert starts == pytest.approx([1.0, 3.0, 9.0], abs=0.05)
    assert cl.segments[0].start == pytest.approx(5.0, abs=0.05)
    # silence is never sent: 12 s of operator channel become two windows of speech
    assert op.windows == 2 and op.audio_seconds < 6.0
    assert all(w <= FakeEngine.max_window for w in engine.windows)
    assert tr.timings["rtf"] >= 0


def test_http_job_roundtrip(tmp_path: Path, engine: FakeEngine, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(app_module.REGISTRY, "fake", ("unused", "unused", {}))
    wav = tmp_path / "call.wav"
    stereo_call(wav, left=[(0.5, 1.0)], right=[(2.0, 2.5)], total=3.0)
    client = TestClient(app_module.app)

    with wav.open("rb") as f:
        job = client.post(
            "/jobs/transcribe",
            files={"file": ("call.wav", f)},
            data={"engine": "fake", "roles": "operator,client"},
        )
    assert job.status_code == 200
    for _ in range(50):
        state = client.get(f"/jobs/{job.json()['id']}").json()
        if state["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["status"] == "done", state["error"]
    assert [c["role"] for c in state["result"]["channels"]] == ["operator", "client"]

    assert (
        client.post("/jobs/transcribe", files={"file": ("x.wav", b"")}, data={"engine": "nope"}).status_code
        == 400
    )
    assert client.get("/jobs/missing").status_code == 404


def test_email_endpoint():
    demo = Path(__file__).resolve().parents[2] / "data" / "cases" / "demo" / "email.eml"
    resp = TestClient(app_module.app).post("/email/clean", content=demo.read_bytes())
    assert resp.status_code == 200
    assert resp.json()["body"].startswith("Здравствуйте, Сергей Викторович!")
