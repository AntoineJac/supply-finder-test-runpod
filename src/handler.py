"""
RunPod Load Balancing Worker — Embeddings + Reranker
Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker

Endpoints
---------
GET  /ping          → Health check (required by RunPod LB)
GET  /              → API info + readiness
GET  /v1/models     → List available models
POST /v1/embeddings → Sentence embeddings (OpenAI-compat)
POST /v1/rerank     → Cross-encoder rerank scores
GET  /stats         → Live server stats
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from sentence_transformers import CrossEncoder, SentenceTransformer

from models import (
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nlp-worker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME  = os.getenv("EMBED_MODEL_NAME",  "/app/.cache/sentence_transformers/paraphrase-multilingual-mpnet-base-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "/app/.cache/sentence_transformers/ms-marco-MiniLM-L4-v2")
PORT              = int(os.getenv("PORT", 80))
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
embed_model:   Optional[SentenceTransformer] = None
rerank_model:  Optional[CrossEncoder]        = None
models_ready:  bool  = False
start_time:    float = time.time()
request_stats: dict  = {"embeddings": 0, "rerank": 0, "errors": 0}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global embed_model, rerank_model, models_ready
    loop = asyncio.get_event_loop()

    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = await loop.run_in_executor(
        None, lambda: SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    )

    logger.info(f"Loading reranker model: {RERANK_MODEL_NAME}")
    rerank_model = await loop.run_in_executor(
        None, lambda: CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
    )

    models_ready = True
    logger.info("✓ Models ready.")

    yield

    logger.info("Shutting down.")
    models_ready = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NLP Inference Worker",
    version="1.0.0",
    description="RunPod Load Balancer worker — embeddings + reranker",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# /ping — REQUIRED by RunPod load balancer
# 200 = healthy  |  204 = still initialising
# ---------------------------------------------------------------------------
@app.get("/ping")
async def health_check():
    if models_ready:
        return {"status": "healthy"}
    return JSONResponse(
        content={"status": "initializing"},
        status_code=status.HTTP_204_NO_CONTENT,
    )


@app.get("/")
async def root():
    return {
        "service": "NLP Inference Worker",
        "ready": models_ready,
        "models": {
            "embeddings": EMBED_MODEL_NAME,
            "reranker": RERANK_MODEL_NAME,
        },
        "endpoints": {
            "health":     "GET  /ping",
            "models":     "GET  /v1/models",
            "embeddings": "POST /v1/embeddings",
            "rerank":     "POST /v1/rerank",
            "stats":      "GET  /stats",
        },
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": EMBED_MODEL_NAME,  "object": "model", "created": now, "owned_by": "sentence-transformers"},
            {"id": RERANK_MODEL_NAME, "object": "model", "created": now, "owned_by": "cross-encoder"},
        ],
    }


@app.get("/stats")
async def stats():
    return {
        "uptime_seconds": int(time.time() - start_time),
        "models_ready": models_ready,
        "device": DEVICE,
        "request_counts": request_stats,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_models():
    if not models_ready:
        request_stats["errors"] += 1
        raise HTTPException(status_code=503, detail="Models not ready — check /ping")


# ---------------------------------------------------------------------------
# POST /v1/embeddings
# ---------------------------------------------------------------------------
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(req: EmbeddingRequest):
    _check_models()
    request_stats["embeddings"] += 1
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    loop = asyncio.get_event_loop()
    vecs: np.ndarray = await loop.run_in_executor(
        None,
        lambda: embed_model.encode(
            texts, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ),
    )
    total = sum(len(t.split()) for t in texts)
    return EmbeddingResponse(
        model=req.model or EMBED_MODEL_NAME,
        data=[{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
        usage={"prompt_tokens": total, "total_tokens": total},
    )


# ---------------------------------------------------------------------------
# POST /v1/rerank
# ---------------------------------------------------------------------------
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    _check_models()
    request_stats["rerank"] += 1
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    pairs = [[req.query, doc] for doc in req.documents]
    loop  = asyncio.get_event_loop()
    scores: np.ndarray = await loop.run_in_executor(
        None, lambda: rerank_model.predict(pairs, convert_to_numpy=True)
    )

    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
    top_n  = req.top_n or len(ranked)
    return RerankResponse(
        model=req.model or RERANK_MODEL_NAME,
        results=[
            {"index": idx, "score": float(s), "document": req.documents[idx] if req.return_documents else None}
            for idx, s in ranked[:top_n]
        ],
        usage={"prompt_tokens": sum(len(d.split()) for d in req.documents)},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")