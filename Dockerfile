# NLP Inference Worker — Embeddings + Reranker (GPU/CUDA)
# Build: docker build --platform linux/amd64 -t yourname/nlp-worker:v1.0 .
# Push:  docker push yourname/nlp-worker:v1.0
#
# pytorch/pytorch base already ships Python 3.11 + torch 2.6 + CUDA 12.4
# so we skip the ~800 MB torch download entirely.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

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

RUN python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)" && \
    python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"

# Bake models into the image — FlashBoot snapshots this layer
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${EMBED_MODEL}'); print('embed model ready')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${RERANK_MODEL}'); print('rerank model ready')"

COPY src /src
ENV PYTHONPATH="/src"
WORKDIR /src

EXPOSE 80
CMD ["python", "-u", "handler.py"]