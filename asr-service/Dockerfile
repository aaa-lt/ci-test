# CPU image for a Hugging Face Docker Space (2 vCPU, 16 GB). On the Mac the service runs
# on the host instead (NFR-28); this image exists because weights cannot be downloaded
# locally during the spikes.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git build-essential cmake zlib1g-dev libbz2-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH VIRTUAL_ENV=/home/user/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
WORKDIR /home/user/app

RUN uv venv $VIRTUAL_ENV
# CPU wheels first, so the engines' torch requirement is already satisfied below.
RUN uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user . .
# tone -> pyctcdecode pins numpy<2, gigaam pins numpy 2 (see pyproject.toml).
RUN echo "numpy>=2" > /tmp/overrides.txt \
    && uv pip install --override /tmp/overrides.txt ".[gigaam,tone,whisper,bench]"

# Bake the default weights into the image so a restarted Space does not download again.
RUN python -c "import gigaam; gigaam.load_model('v3_rnnt', fp16_encoder=False, device='cpu')" \
    && python -c "from tone import StreamingCTCPipeline; StreamingCTCPipeline.from_hugging_face()" \
    && python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-large-v3-turbo')"

EXPOSE 7860
CMD ["uvicorn", "asr_service.app:app", "--host", "0.0.0.0", "--port", "7860"]
