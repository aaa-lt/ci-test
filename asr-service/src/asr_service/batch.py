"""Offline batch runs without the HTTP layer, for CI runners and the host.

    python -m asr_service.batch transcribe --audio DIR --out DIR --engines a,b
    python -m asr_service.batch bench --engines a,b --n 150 --out FILE

`transcribe` takes every <call>.wav with a <call>.reference.json next to it (for the
channel roles) and writes <out>/<call>.<engine>.json in the same format as the HTTP API.
Existing outputs are kept, so a rerun only does what is missing (FR-25).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_service.transcribe import transcribe_file


def transcribe(audio: Path, out: Path, engines: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for engine in engines:
        for wav in sorted(audio.glob("*.wav")):
            target = out / f"{wav.stem}.{engine}.json"
            ref = wav.with_suffix(".reference.json")
            if target.exists() or not ref.exists():
                continue
            roles = json.loads(ref.read_text(encoding="utf-8"))["channels"]
            result = transcribe_file(wav, engine, roles)
            target.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"{wav.stem} {engine}: rtf {result.timings['rtf']}, load {result.timings['load']} s",
                flush=True,
            )


def bench(engines: list[str], n: int, out: Path, force: bool = False) -> None:
    from asr_service import bench as bench_module  # needs the "bench" extra

    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    todo = [
        e
        for e in engines
        if force or not existing or e not in existing["engines"] or existing["samples"] != n
    ]
    if not todo:
        return
    report = bench_module.run(todo, n)
    if existing and existing["samples"] == n:
        report["engines"] = existing["engines"] | report["engines"]
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, r in report["engines"].items():
        print(f"{name}: WER {r['wer']:.3f}, rtf {r['rtf']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="asr_service.batch")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("transcribe")
    t.add_argument("--audio", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--engines", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--engines", required=True)
    b.add_argument("--n", type=int, default=150)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--force", action="store_true", help="rerun engines already in the report")
    args = parser.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if args.command == "transcribe":
        transcribe(args.audio, args.out, engines)
    else:
        bench(engines, args.n, args.out, args.force)


if __name__ == "__main__":
    main()
