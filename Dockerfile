# NLP Inference Worker — Embeddings + Reranker (GPU/CUDA)
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0
#
# Models are NOT baked in. On first cold start they download to the RunPod
# network volume (/runpod-volume) and are cached there for all future starts.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── System packages ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        curl \
        libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && pip install --no-cache-dir --upgrade pip uv \
    && rm -rf /var/lib/apt/lists/*

# ── Environment ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/runpod-volume \
    # Prevent uv from re-downloading torch on every build when cache is warm
    UV_SYSTEM_PYTHON=1

# ── Heavy deps (torch ~800 MB) — own layer, rebuilt only on version change ─
# IMPORTANT: torch cu124 must install BEFORE infinity-emb[torch] so that
# infinity-emb reuses this build instead of pulling a CPU wheel from PyPI.
COPY requirements-heavy.txt /requirements-heavy.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements-heavy.txt

# ── Light deps ─────────────────────────────────────────────────────────────
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# ── Smoke-test: verify torch sees CUDA and infinity-emb imports cleanly ───
# This runs at BUILD time — catches missing libs before the image is pushed.
RUN python -c "
import torch
print('torch:', torch.__version__)
print('cuda available (build-time check):', torch.cuda.is_available())
from infinity_emb.engine import AsyncEngineArray, EngineArgs
from sentence_transformers import SentenceTransformer
print('infinity-emb + sentence-transformers OK')
"

# ── App code ───────────────────────────────────────────────────────────────
COPY src /src
ENV PYTHONPATH="/src"
WORKDIR /src

EXPOSE 80

CMD ["python", "-u", "handler.py"]