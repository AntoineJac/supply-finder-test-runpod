# NLP Inference Worker — Embeddings + Reranker (CPU)
# Build: docker build -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0

FROM python:3.11-slim

ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L4-v2

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    EMBED_MODEL_NAME=${EMBED_MODEL} \
    RERANK_MODEL_NAME=${RERANK_MODEL}

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"

# Bake models into the image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${EMBED_MODEL}'); print('embed model ready')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${RERANK_MODEL}'); print('rerank model ready')"

COPY src /src
ENV PYTHONPATH="/src"
WORKDIR /src

EXPOSE 80
CMD ["python", "-u", "handler.py"]