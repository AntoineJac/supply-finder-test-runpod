# NLP Inference Worker — Embeddings + Reranker (CPU)
# Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker
#
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0

# ============================
# Stage 1: Model downloader
# ============================
# Uses .save() so only inference files land in the image — no HF blob cache overhead.
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir sentence-transformers==2.7.0

RUN mkdir -p /app/.cache/sentence_transformers && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2').save('/app/.cache/sentence_transformers/paraphrase-multilingual-mpnet-base-v2')" && \
    python -c "from sentence_transformers.cross_encoder import CrossEncoder; m = CrossEncoder('cross-encoder/ms-marco-MiniLM-L4-v2'); m.save('/app/.cache/sentence_transformers/ms-marco-MiniLM-L4-v2')"

# ============================
# Stage 2: Builder
# ============================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# Heavy deps first (torch ~800 MB) — layer only rebuilt when versions change.
# --mount=type=cache persists the pip wheel cache on the build host across runs.
COPY requirements-heavy.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-heavy.txt

# Light deps — fast to reinstall; torch layer stays cached above.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ============================
# Stage 3: Runtime
# ============================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    HF_HOME=/app/.cache/sentence_transformers

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy pre-saved models (clean flat directories, no blob cache)
COPY --from=model-downloader /app/.cache/sentence_transformers /app/.cache/sentence_transformers

# App code
COPY src /src
ENV PYTHONPATH="/:/src"
WORKDIR /src

EXPOSE 80

CMD ["python3", "handler.py"]