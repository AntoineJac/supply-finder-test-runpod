# GLM-4.7-Flash RunPod Load Balancing Worker
# Follows: https://docs.runpod.io/serverless/load-balancing/vllm-worker
#
# Build: docker build --platform linux/amd64 -t yourname/glm47-worker:v2.1 .
# Push:  docker push yourname/glm47-worker:v2.1

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
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

# ── vLLM nightly — GLM-4.7-Flash only works on nightly, not stable release ──
# Per zai-org HF model card: must use pypi.org as primary index-url
RUN pip install --no-cache-dir \
    --upgrade \
    --pre \
    vllm \
    --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/nightly

# ── Transformers from source (required for GLM-4.7-Flash) ───────────────────
RUN pip install --no-cache-dir --upgrade --force-reinstall \
    "git+https://github.com/huggingface/transformers.git"

RUN pip install --no-cache-dir numba accelerate

# ── Pre-download aux models at build time (zero cold-start for embeddings) ───
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