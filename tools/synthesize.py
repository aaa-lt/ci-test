# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "numpy>=2"]
# ///
"""Voice call scripts with Yandex SpeechKit and assemble telephone-quality stereo calls.

Usage (from the repository root):
    uv run tools/synthesize.py                 # every data/scripts/*.txt
    uv run tools/synthesize.py demo-call       # one script by name

For each script writes data/audio/<name>.wav (8 kHz, 16-bit, operator left, client
right, band-limited and passed through G.711 mu-law) and data/audio/<name>.reference.json
with the exact text and placement of every utterance. Raw synthesis is cached in
data/audio/raw/ keyed by voice and text, so re-running only pays for changed lines.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "data" / "scripts"
AUDIO = ROOT / "data" / "audio"
RAW = AUDIO / "raw"

TTS_URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
SYNTH_RATE = 16000
LEAD_IN = 0.8  # silence before the first utterance, seconds
NOISE_DBFS = -50.0  # line noise floor added before the telephone chain
SEED = 7

ENTITY = re.compile(r"\[\[(.+?)\|(\w+)\]\]")
LINE = re.compile(r"^(?P<role>[OC])(?:\((?P<op>[+~])(?P<sec>\d+(?:\.\d+)?)\))?:\s*(?P<text>.+)$")
ROLES = {"O": "operator", "C": "client"}


@dataclass
class Entity:
    text: str
    type: str


@dataclass
class Utterance:
    role: str
    text: str
    entities: list[Entity]
    gap: float | None  # explicit gap after the previous utterance; negative means overlap
    start: float = 0.0
    end: float = 0.0


@dataclass
class Script:
    name: str
    voices: dict[str, tuple[str, str | None]]  # role -> (voice, emotion)
    pause: float
    utterances: list[Utterance] = field(default_factory=list)


def parse_script(path: Path) -> Script:
    """Parse the call script format described in data/README.md."""
    voices: dict[str, tuple[str, str | None]] = {}
    pause = 0.4
    utterances: list[Utterance] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            directive, _, rest = line.lstrip("# ").partition(" ")
            if directive == "voice":
                for pair in rest.split():
                    role, _, spec = pair.partition("=")
                    voice, _, emotion = spec.partition(":")
                    voices[role] = (voice, emotion or None)
            elif directive == "pause":
                pause = float(rest)
            continue
        m = LINE.match(line)
        if not m:
            raise ValueError(f"{path.name}: cannot parse line: {line}")
        body = m["text"]
        entities = [Entity(text=e[1], type=e[2]) for e in ENTITY.finditer(body)]
        gap = None
        if m["op"]:
            gap = float(m["sec"]) * (1 if m["op"] == "+" else -1)
        utterances.append(Utterance(ROLES[m["role"]], ENTITY.sub(r"\1", body), entities, gap))
    missing = {"operator", "client"} - voices.keys()
    if missing:
        raise ValueError(f"{path.name}: no voice for {missing}")
    return Script(path.stem, voices, pause, utterances)


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#") and value.strip():
                env.setdefault(key.strip(), value.strip())
    return env


def synthesize(client: httpx.Client, text: str, voice: str, emotion: str | None, folder: str) -> np.ndarray:
    """Return mono float32 samples at SYNTH_RATE for one utterance, using the on-disk cache."""
    key = hashlib.sha256(f"{voice}|{emotion}|{text}".encode()).hexdigest()[:20]
    cached = RAW / f"{key}.pcm"
    if not cached.exists():
        form = {
            "text": text,
            "lang": "ru-RU",
            "voice": voice,
            "format": "lpcm",
            "sampleRateHertz": str(SYNTH_RATE),
            "folderId": folder,
        }
        if emotion:
            form["emotion"] = emotion
        resp = client.post(TTS_URL, data=form)
        if resp.status_code != 200:
            raise RuntimeError(f"TTS failed ({resp.status_code}): {resp.text[:300]}")
        RAW.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(resp.content)
    pcm = np.frombuffer(cached.read_bytes(), dtype="<i2")
    return trim_silence(pcm.astype(np.float32) / 32768.0)


def trim_silence(x: np.ndarray, threshold: float = 0.01, keep: float = 0.05) -> np.ndarray:
    """Drop leading and trailing near-silence so that scripted gaps are the real gaps."""
    loud = np.flatnonzero(np.abs(x) > threshold)
    if loud.size == 0:
        return x
    pad = int(keep * SYNTH_RATE)
    return x[max(0, loud[0] - pad) : min(len(x), loud[-1] + pad)]


def assemble(script: Script, clips: list[np.ndarray]) -> np.ndarray:
    """Place clips on a two-channel timeline and fill in utterance start and end times."""
    rng = np.random.default_rng(SEED)
    cursor = LEAD_IN
    for utt, clip in zip(script.utterances, clips, strict=True):
        if utt is script.utterances[0]:
            gap = 0.0
        elif utt.gap is not None:
            gap = utt.gap
        else:
            gap = script.pause * float(rng.uniform(0.7, 1.4))
        utt.start = max(0.0, round(cursor + gap, 3))
        utt.end = round(utt.start + len(clip) / SYNTH_RATE, 3)
        cursor = max(cursor, utt.end)
    total = int((cursor + 0.8) * SYNTH_RATE)
    stereo = np.zeros((total, 2), dtype=np.float32)
    for utt, clip in zip(script.utterances, clips, strict=True):
        ch = 0 if utt.role == "operator" else 1
        i = int(utt.start * SYNTH_RATE)
        stereo[i : i + len(clip), ch] += clip
    noise = rng.normal(0.0, 10 ** (NOISE_DBFS / 20), stereo.shape).astype(np.float32)
    return np.clip(stereo + noise, -1.0, 1.0)


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(samples.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((samples * 32767).astype("<i2").tobytes())


def telephone(src: Path, dst: Path) -> None:
    """Band-limit to 300-3400 Hz, resample to 8 kHz and round-trip through G.711 mu-law."""
    mulaw = dst.with_suffix(".mulaw.wav")
    ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run(
        [*ff, "-i", str(src), "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw", str(mulaw)],
        check=True,
    )
    subprocess.run([*ff, "-i", str(mulaw), "-c:a", "pcm_s16le", str(dst)], check=True)
    mulaw.unlink()


def main(names: list[str]) -> None:
    env = load_env()
    key, folder = env.get("YANDEX_API_KEY"), env.get("YANDEX_FOLDER_ID")
    if not key or not folder:
        sys.exit("YANDEX_API_KEY and YANDEX_FOLDER_ID must be set in .env")
    paths = [SCRIPTS / f"{n}.txt" for n in names] if names else sorted(SCRIPTS.glob("*.txt"))
    AUDIO.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"Authorization": f"Api-Key {key}"}, timeout=60) as client:
        for path in paths:
            script = parse_script(path)
            clips = [synthesize(client, u.text, *script.voices[u.role], folder) for u in script.utterances]
            stereo = assemble(script, clips)
            studio = RAW / f"{script.name}.studio.wav"
            write_wav(studio, stereo, SYNTH_RATE)
            telephone(studio, AUDIO / f"{script.name}.wav")
            reference = {
                "call": script.name,
                "sampleRate": 8000,
                "channels": ["operator", "client"],
                "duration": round(len(stereo) / SYNTH_RATE, 3),
                "voices": {r: {"voice": v, "emotion": e} for r, (v, e) in script.voices.items()},
                "utterances": [{k: v for k, v in asdict(u).items() if k != "gap"} for u in script.utterances],
            }
            (AUDIO / f"{script.name}.reference.json").write_text(
                json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"{script.name}: {len(script.utterances)} utterances, {reference['duration']:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
