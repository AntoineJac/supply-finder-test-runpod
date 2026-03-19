# NLP Inference Worker — Embeddings + Reranker (GPU/CUDA)
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0
#
# Models are baked into the image so FlashBoot can snapshot them.
# Cold starts only need to load weights into GPU memory (~2-3 s), not download.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── Build args — override with --build-arg to change models ───────────────
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L4-v2

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
    # Models live inside the image — FlashBoot snapshots them
    HF_HOME=/models \
    # Expose model names as runtime env so handler.py defaults match
    EMBED_MODEL_NAME=${EMBED_MODEL} \
    RERANK_MODEL_NAME=${RERANK_MODEL} \
    UV_SYSTEM_PYTHON=1

# ── Heavy deps (torch ~800 MB) — own layer, rebuilt only on version change ─
COPY requirements-heavy.txt /requirements-heavy.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements-heavy.txt

# ── Light deps ─────────────────────────────────────────────────────────────
COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# ── Smoke-test ────────────────────────────────────────────────────────────
RUN python -c "import torch; print('torch:', torch.__version__)" && \
    python -c "from infinity_emb.engine import AsyncEngineArray, EngineArgs; print('infinity-emb OK')"

# ── Bake models into the image layer ─────────────────────────────────────
# Downloaded to HF_HOME=/models at build time.
# FlashBoot snapshots this layer → cold start = GPU load only, no network I/O.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${EMBED_MODEL}'); print('embed model ready')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${RERANK_MODEL}'); print('rerank model ready')"

# ── App code ───────────────────────────────────────────────────────────────
COPY src /src
ENV PYTHONPATH="/src"
WORKDIR /src

EXPOSE 80

CMD ["python", "-u", "handler.py"]