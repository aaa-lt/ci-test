"""Open STT shard parsing, on a tiny WebDataset tar of the same layout."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from asr_service import bench


def add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_load_samples_pairs_wav_and_json_and_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    shard = tmp_path / "validate-00000.tar"
    with tarfile.open(shard, "w") as tar:
        for key, duration, text in [
            ("a1", 0.5, "коротко"),
            ("b2", 2.0, "два слова"),
            ("c3", 1.5, " "),
            ("d4", 3.0, "три"),
        ]:
            add(tar, f"{key}.json", json.dumps({"id": key, "duration": duration, "text": text}).encode())
            add(tar, f"{key}.wav", b"RIFF" + key.encode())
    monkeypatch.setattr(bench, "hf_hub_download", lambda *a, **k: str(shard))

    samples = bench.load_samples(n=5, min_duration=1.0)
    assert [(s["id"], s["wav"]) for s in samples] == [("b2", b"RIFFb2"), ("d4", b"RIFFd4")]
    assert bench.load_samples(n=1)[0]["id"] == "b2"
