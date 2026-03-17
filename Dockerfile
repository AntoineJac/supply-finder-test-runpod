# ── RunPod Load Balancing Worker — GLM-4.7-Flash + Embeddings + Reranker ──
# Follows: https://docs.runpod.io/serverless/load-balancing/vllm-worker
#
# Build:
#   docker build --platform linux/amd64 -t yourname/glm47-worker:v1.0 .
# Push:
#   docker push yourname/glm47-worker:v1.0

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# ── System deps ──────────────────────────────────────────────────────────────
# python3.12 is not in Ubuntu 22.04 default repos — use 3.11 (available natively)
RUN apt-get update -y && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git curl wget build-essential \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.4/compat/ 2>/dev/null || true

# ── Python deps: base ────────────────────────────────────────────────────────
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# ── vLLM nightly (required for GLM-4.7-Flash) ───────────────────────────────
RUN pip install --no-cache-dir \
    --upgrade \
    vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly

# ── Transformers from source (required for GLM-4.7-Flash) ───────────────────
RUN pip install --no-cache-dir --upgrade --force-reinstall \
    "git+https://github.com/huggingface/transformers.git"

RUN pip install --no-cache-dir numba accelerate

# ── Pre-download aux models at build time (baked in, no runtime delay) ───────
RUN python3 -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L4-v2') \
"

# ── App code ─────────────────────────────────────────────────────────────────
COPY src /src
ENV PYTHONPATH="/:/src"

WORKDIR /src

# Port 80 is the RunPod load balancer default (set via PORT env var)
EXPOSE 80

CMD ["python3", "handler.py"]