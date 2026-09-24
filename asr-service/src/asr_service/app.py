"""HTTP API of the speech service.

Recognition takes minutes on CPU, longer than proxies keep a request open, so it runs
as a job: POST returns an id, GET /jobs/{id} returns the status and, when done, the result.
One worker thread runs jobs in order; engines are CPU-bound and would only contend.
"""

from __future__ import annotations

import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from asr_service.email_clean import CleanEmail, clean_email
from asr_service.engines import REGISTRY, available
from asr_service.transcribe import transcribe_file

app = FastAPI(title="talkodex-asr")
executor = ThreadPoolExecutor(max_workers=1)


class Job(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    created: float
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str | None = None


jobs: dict[str, Job] = {}


def submit(kind: str, fn: Any, *args: Any) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, created=time.time())
    jobs[job.id] = job

    def run() -> None:
        job.status, job.started = "running", time.time()
        try:
            job.result = fn(*args)
            job.status = "done"
        except Exception:  # noqa: BLE001 - the job record is the only place a failure can be reported
            job.error = traceback.format_exc(limit=5)
            job.status = "failed"
        job.finished = time.time()

    executor.submit(run)
    return job


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "engines": available(),
        "jobs": {s: sum(j.status == s for j in jobs.values()) for s in ("queued", "running")},
    }


@app.post("/jobs/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    engine: str = Form(...),
    roles: str | None = Form(None, description="comma-separated role per channel, e.g. operator,client"),
) -> Job:
    if engine not in REGISTRY:
        raise HTTPException(400, f"unknown engine {engine}")
    tmp = Path(tempfile.mkdtemp())
    path = tmp / (file.filename or "audio")
    path.write_bytes(await file.read())
    role_list = [r.strip() or None for r in roles.split(",")] if roles else None

    def work() -> dict[str, Any]:
        try:
            return transcribe_file(path, engine, role_list).model_dump()
        finally:
            path.unlink(missing_ok=True)
            tmp.rmdir()

    return submit("transcribe", work)


class BenchRequest(BaseModel):
    engines: list[str]
    n: int = 150


@app.post("/jobs/bench")
def bench(req: BenchRequest) -> Job:
    from asr_service import bench as bench_module  # needs the "bench" extra

    return submit("bench", bench_module.run, req.engines, req.n)


@app.get("/jobs/{job_id}")
def job(job_id: str) -> Job:
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return jobs[job_id]


@app.post("/email/clean")
async def email(request: Request) -> CleanEmail:
    """Body: the raw RFC 822 message."""
    try:
        return clean_email(await request.body())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
