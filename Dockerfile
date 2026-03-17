# NLP Inference Worker — Embeddings + Reranker (CPU)
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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