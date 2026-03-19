# NLP Inference Worker — Embeddings + Reranker (GPU/CUDA)
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L4-v2

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && pip install --no-cache-dir --upgrade pip uv \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    EMBED_MODEL_NAME=${EMBED_MODEL} \
    RERANK_MODEL_NAME=${RERANK_MODEL} \
    UV_SYSTEM_PYTHON=1

# Heavy deps (torch ~800 MB) — own layer for cache efficiency
COPY requirements-heavy.txt /requirements-heavy.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements-heavy.txt

COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

RUN python -c "import torch; print('torch:', torch.__version__)" && \
    python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"

# Bake models into the image — FlashBoot snapshots this layer
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${EMBED_MODEL}'); print('embed model ready')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${RERANK_MODEL}'); print('rerank model ready')"

COPY src /src
ENV PYTHONPATH="/src"
WORKDIR /src

EXPOSE 80
CMD ["python", "-u", "handler.py"]