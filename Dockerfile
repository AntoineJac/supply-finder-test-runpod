# NLP Inference Worker — Embeddings + Reranker (GPU/CUDA)
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0
#
# Models are NOT baked into the image. On first cold start they download to the
# RunPod network volume (/runpod-volume) and are cached there for all future starts.

# ============================
# Stage 1: Builder
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
# Stage 2: Runtime
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
    HF_HOME=/runpod-volume

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# App code
COPY src /src
ENV PYTHONPATH="/:/src"
WORKDIR /src

EXPOSE 80

CMD ["/opt/venv/bin/python", "handler.py"]
