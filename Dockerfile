# NLP Inference Worker — Embeddings + Reranker (GPU/CUDA)
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0

# ============================
# Stage 1: Model downloader
# ============================
# snapshot_download stores models in standard HF cache format,
# which infinity-emb (and transformers) reads natively — no blob overhead tricks needed.
FROM python:3.11-slim AS model-downloader

ENV HF_HOME=/app/.cache/huggingface

RUN pip install --no-cache-dir huggingface-hub

RUN mkdir -p /app/.cache/huggingface && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*','rust_model*'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/ms-marco-MiniLM-L4-v2', ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*','rust_model*'])"

# ============================
# Stage 2: Builder
# ============================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1

# uv is significantly faster than pip for dependency resolution + install
RUN pip install --no-cache-dir uv

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Heavy deps first (torch ~800 MB) — layer only rebuilt when versions change.
# --mount=type=cache keeps the uv wheel cache on the build host across runs.
COPY requirements-heavy.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements-heavy.txt

# Light deps — fast to reinstall; torch/infinity-emb layer stays cached above.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# ============================
# Stage 3: Runtime
# ============================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python 3.11 + symlink so `python` works
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    git \
    wget \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy pre-downloaded models (HF cache format — infinity-emb finds them by model ID)
COPY --from=model-downloader /app/.cache/huggingface /app/.cache/huggingface

# App code
COPY src /src
ENV PYTHONPATH="/:/src"
WORKDIR /src

EXPOSE 80

CMD ["python", "handler.py"]