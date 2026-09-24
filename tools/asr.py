# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "huggingface_hub>=0.34", "jiwer>=3", "num2words>=0.5"]
# ///
"""Drive asr-service: deploy the Space, transcribe the synthetic calls, benchmark, score.

    uv run tools/asr.py deploy                    create or update the private Space
    uv run tools/asr.py health
    uv run tools/asr.py transcribe [engines...]   data/audio/*.wav -> results/transcripts/
    uv run tools/asr.py bench [engines...]        Open STT WER inside the Space -> results/asr/
    uv run tools/asr.py score                     transcripts vs references -> results/asr/calls.json

Local engines run in the Space (ASR_URL), the cloud engine through the service on this
machine (ASR_LOCAL_URL); requests to hf.space carry HF_TOKEN.
Transcripts are cached per call and engine and reused (FR-25); delete a file to redo it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
import jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "asr-service" / "src"))
from asr_service.textnorm import normalize  # noqa: E402

AUDIO = ROOT / "data" / "audio"
TRANSCRIPTS = ROOT / "results" / "transcripts"
RESULTS = ROOT / "results" / "asr"
SPACE_NAME = "talkodex-asr"
LOCAL_ENGINES = ["gigaam-rnnt", "tone", "tone-greedy", "whisper-turbo"]
CLOUD_ENGINES = {"yandex"}


def env() -> dict[str, str]:
    values = dict(os.environ)
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#") and value.strip():
                values.setdefault(key.strip(), value.strip())
    return values


def client(engine: str | None = None) -> httpx.Client:
    """The Space for local engines, the service on this machine for the cloud engine."""
    e = env()
    local = e.get("ASR_LOCAL_URL", "http://127.0.0.1:8000")
    url = local if engine in CLOUD_ENGINES else e.get("ASR_URL", local)
    headers = {"Authorization": f"Bearer {e['HF_TOKEN']}"} if "hf.space" in url and e.get("HF_TOKEN") else {}
    return httpx.Client(base_url=url, headers=headers, timeout=120)


def wait(c: httpx.Client, job_id: str, poll: float = 5.0) -> Any:
    while True:
        job = c.get(f"/jobs/{job_id}").raise_for_status().json()
        if job["status"] == "done":
            return job["result"]
        if job["status"] == "failed":
            raise RuntimeError(job["error"])
        time.sleep(poll)


def deploy() -> None:
    from huggingface_hub import HfApi

    e = env()
    api = HfApi(token=e["HF_TOKEN"])
    repo = f"{e['HF_SPACE_OWNER']}/{SPACE_NAME}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="space",
        folder_path=str(ROOT / "asr-service"),
        ignore_patterns=[".venv/*", "**/__pycache__/*", ".pytest_cache/*", ".ruff_cache/*"],
    )
    print(f"https://huggingface.co/spaces/{repo}  host: https://{api.space_info(repo).host}")


def transcribe(engines: list[str]) -> None:
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    for engine in engines:
        with client(engine) as c:
            for wav in sorted(AUDIO.glob("*.wav")):
                out = TRANSCRIPTS / f"{wav.stem}.{engine}.json"
                if out.exists():
                    continue
                with wav.open("rb") as f:
                    job = (
                        c.post(
                            "/jobs/transcribe",
                            files={"file": (wav.name, f, "audio/wav")},
                            data={"engine": engine, "roles": "operator,client"},
                        )
                        .raise_for_status()
                        .json()
                    )
                result = wait(c, job["id"])
                out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{wav.stem} {engine}: rtf {result['timings']['rtf']}, total {result['timings']['total']} s")


def bench(engines: list[str], n: int = 150) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with client() as c:
        job = c.post("/jobs/bench", json={"engines": engines, "n": n}).raise_for_status().json()
        report = wait(c, job["id"], poll=15)
    out = RESULTS / "openstt.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, r in report["engines"].items():
        print(f"{name}: WER {r['wer']:.3f}, rtf {r['rtf']}")


def score() -> None:
    """Per call and engine: WER per channel, entity recall by type, utterance onset error."""
    report: dict[str, Any] = {}
    for ref_path in sorted(AUDIO.glob("*.reference.json")):
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        call = ref["call"]
        for tr_path in sorted(TRANSCRIPTS.glob(f"{call}.*.json")):
            engine = tr_path.name.removeprefix(f"{call}.").removesuffix(".json")
            tr = json.loads(tr_path.read_text(encoding="utf-8"))
            report.setdefault(engine, {})[call] = score_call(ref, tr)
    summary = {}
    for engine, calls in report.items():
        refs = [r for c in calls.values() for r in c.pop("_ref")]
        hyps = [h for c in calls.values() for h in c.pop("_hyp")]
        onsets = [c["onset_mean"] for c in calls.values() if c["onset_mean"] is not None]
        summary[engine] = {
            "wer": round(jiwer.wer(refs, hyps), 4),
            "rtf": round(mean(c["rtf"] for c in calls.values()), 4),
            "onset_mean": round(mean(onsets), 3) if onsets else None,
            "entities": merge_entities([c["entities"] for c in calls.values()]),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "calls": report}
    (RESULTS / "calls.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for engine, s in summary.items():
        print(f"{engine}: WER {s['wer']:.3f}, onset {s['onset_mean']:.2f} s, entities {s['entities']}")


def score_call(ref: dict[str, Any], tr: dict[str, Any]) -> dict[str, Any]:
    roles = {ch["role"]: ch for ch in tr["channels"]}
    refs, hyps, onsets = [], [], []
    entities: dict[str, list[int]] = {}
    for role in ("operator", "client"):
        utts = [u for u in ref["utterances"] if u["role"] == role]
        segs = roles[role]["segments"]
        refs.append(normalize(" ".join(u["text"] for u in utts)))
        hyps.append(normalize(" ".join(s["text"] for s in segs)))
        for u in utts:
            near = [s for s in segs if s["start"] < u["end"] + 1.0 and s["end"] > u["start"] - 1.0]
            if near:
                onsets.append(min(abs(s["start"] - u["start"]) for s in near))
            local = normalize(" ".join(s["text"] for s in near))
            for ent in u["entities"]:
                hit = f" {normalize(ent['text'])} " in f" {local} "
                entities.setdefault(ent["type"], [0, 0])
                entities[ent["type"]][0] += hit
                entities[ent["type"]][1] += 1
    return {
        "wer": round(jiwer.wer(refs, hyps), 4),
        "rtf": tr["timings"]["rtf"],
        "onset_mean": round(mean(onsets), 3) if onsets else None,
        "onset_p95": round(sorted(onsets)[int(0.95 * (len(onsets) - 1))], 3) if onsets else None,
        "entities": entities,
        "_ref": refs,
        "_hyp": hyps,
    }


def merge_entities(items: list[dict[str, list[int]]]) -> dict[str, str]:
    total: dict[str, list[int]] = {}
    for item in items:
        for k, (hit, n) in item.items():
            t = total.setdefault(k, [0, 0])
            t[0] += hit
            t[1] += n
    return {k: f"{hit}/{n}" for k, (hit, n) in sorted(total.items())}


def main() -> None:
    cmd, *args = sys.argv[1:] or ["health"]
    if cmd == "deploy":
        deploy()
    elif cmd == "health":
        with client() as c:
            print(json.dumps(c.get("/health").raise_for_status().json(), indent=2))
    elif cmd == "transcribe":
        transcribe(args or [*LOCAL_ENGINES, "yandex"])
    elif cmd == "bench":
        bench(args or LOCAL_ENGINES)
    elif cmd == "score":
        score()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
