---
title: talkodex-asr
sdk: docker
app_port: 7860
pinned: false
---

# asr-service

Speech recognition with word timestamps and e-mail cleanup, behind HTTP.

| Engine | Model | Timestamps | Window |
|---|---|---|---|
| `gigaam-rnnt`, `gigaam-ctc` | GigaAM v3, native package | words, from the package | 24 s, VAD cuts |
| `tone` | T-One, beam search with KenLM | phrases only | whole channel |
| `tone-greedy` | T-One, greedy CTC | words, own CTC alignment | whole channel |
| `whisper-turbo` | Whisper large-v3-turbo, transformers on CPU | words | 28 s, VAD cuts |
| `yandex` | SpeechKit v3 async, cloud reference | words | whole channel |

A stereo file is split into channels and each channel is recognized separately. Engines
with a length limit get speech-only windows from `chunking.py` (FR-24).

## API

```
GET  /health                         engines and why an engine is unavailable
POST /jobs/transcribe                multipart: file, engine, roles="operator,client"
POST /jobs/bench                     {"engines": [...], "n": 150}: Open STT phone-call WER
GET  /jobs/{id}                      status, result, error
POST /email/clean                    raw RFC 822 message -> body without quotes and signature
```

## Run

Local, only the cloud engine and e-mail cleanup (no PyTorch):

```bash
uv run --env-file ../.env uvicorn asr_service.app:app --port 8000
```

All engines: the Dockerfile, deployed as a private Space by `tools/deploy_space.py`.

```bash
uv run pytest
```
