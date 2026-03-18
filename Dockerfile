# NLP Inference Worker — Embeddings + Reranker (GPU/CUDA)
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0
#
# Models are NOT baked into the image. On first cold start they download to the
# RunPod network volume (/runpod-volume) and are cached there for all future starts.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python 3.11 + uv + symlinks
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && pip install --no-cache-dir uv \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume

# Heavy deps first (torch ~800 MB) — layer only rebuilt when versions change.
COPY requirements-heavy.txt /requirements-heavy.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements-heavy.txt

# Light deps
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# App code
COPY src /src
ENV PYTHONPATH="/:/src"
WORKDIR /src

EXPOSE 80

CMD ["python", "-u", "handler.py"]
