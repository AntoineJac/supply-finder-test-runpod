# NLP Inference Worker — Embeddings + Reranker
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# RunPod official pattern: just python3-pip, use Ubuntu 22.04's built-in Python 3.10
# No PPA, no deadsnakes, no manual symlinks — same as docs.runpod.io/serverless/load-balancing/vllm-worker
RUN apt-get update -y \
    && apt-get install -y python3-pip git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.4/compat/ 2>/dev/null || true

# ── Base Python deps ─────────────────────────────────────────────────────────
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# ── Pre-download models at build time (zero cold-start) ─────────────────────
RUN python3 -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L4-v2') \
"

# ── App code ─────────────────────────────────────────────────────────────────
COPY src /src
ENV PYTHONPATH="/:/src"
WORKDIR /src

EXPOSE 80

CMD ["python3", "handler.py"]