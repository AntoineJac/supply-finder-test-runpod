# ─────────────────────────────────────────────────────────────────────────────
# NLP Inference Image
# Base: CUDA 12.4 + Python 3.11 (matches SGLang / vLLM requirements)
#
# Layers:
#   1. System deps
#   2. SGLang (GLM-4.7-Flash LLM inference)
#   3. sentence-transformers (embed + rerank)
#   4. FastAPI app
#
# Build:
#   docker build -t nlp-inference:latest .
#
# Run (single GPU, HF cache on host):
#   docker run --gpus all -p 8000:8000 \
#     -v /workspace:/workspace \
#     -e NLP_API_KEY=secret \
#     -e GLM_MODEL_PATH=zai-org/GLM-4.7-Flash \
#     nlp-inference:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ── System ────────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git curl wget ca-certificates \
        build-essential ninja-build \
        libssl-dev libffi-dev \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3.11 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip + wheel
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# ── SGLang (GLM-4.7-Flash requires nightly build) ────────────────────────────
# GLM-4.7-Flash MoE support is only in recent SGLang; pin to a known-good ver.
RUN pip install --no-cache-dir \
    "sglang[all]" \
    --extra-index-url https://sgl-project.github.io/whl/cu124

# Install transformers from HEAD (GLM-4.7-Flash tokenizer needs latest)
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git

# flashinfer kernel — massive attention speedup for SGLang on CUDA 12.4
RUN pip install --no-cache-dir flashinfer-python \
    -i https://flashinfer.ai/whl/cu124/torch2.4/

# ── sentence-transformers stack ───────────────────────────────────────────────
RUN pip install --no-cache-dir \
    sentence-transformers==3.4.1 \
    torch==2.4.1 \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ── FastAPI + server ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    httpx==0.27.2 \
    pydantic==2.8.2 \
    numpy

# ── Pre-download small models into the image ──────────────────────────────────
# This bakes embed + rerank models into the image so cold starts are instant.
# GLM (~60GB) is too large to bake in — it loads from HF_HOME at runtime.
RUN python3 -c " \
from sentence_transformers import SentenceTransformer; \
from sentence_transformers.cross_encoder import CrossEncoder; \
print('Downloading paraphrase-multilingual-mpnet-base-v2...'); \
SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); \
print('Downloading ms-marco-MiniLM-L4-v2...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L4-v2'); \
print('Done.') \
"

# ── App ───────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 30000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]