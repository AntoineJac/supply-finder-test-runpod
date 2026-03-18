"""
RunPod Load Balancing Worker — Embeddings + Reranker
Follows: https://docs.runpod.io/serverless/load-balancing/build-a-worker

Endpoints
---------
GET  /ping          → Health check (required by RunPod LB)
GET  /              → API info + readiness
GET  /v1/models     → List loaded models
POST /v1/embeddings → Sentence embeddings (OpenAI-compat)
POST /v1/rerank     → Cross-encoder rerank scores
GET  /stats         → Live server stats
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from embedding_service import EmbeddingService
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
EMBED_MODEL  = os.getenv("EMBED_MODEL_NAME",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L4-v2")
PORT         = int(os.getenv("PORT", 80))

# Ensure both default models are loaded by the engine array (can be overridden via MODEL_NAMES)
if not os.environ.get("MODEL_NAMES"):
    os.environ["MODEL_NAMES"] = f"{EMBED_MODEL};{RERANK_MODEL}"

# ---------------------------------------------------------------------------
# Service singleton — fail fast so the container exits cleanly on bad config
# ---------------------------------------------------------------------------
try:
    embedding_service = EmbeddingService()
except Exception as exc:
    sys.stderr.write(f"\nStartup failed: {exc}\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
models_ready:  bool  = False
start_time:    float = time.time()
request_stats: dict  = {"embeddings": 0, "rerank": 0, "errors": 0}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global models_ready
    await embedding_service.start()
    models_ready = True
    yield
    models_ready = False
    await embedding_service.stop()
    logger.info("Shutting down.")


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
        "models": embedding_service.list_models() if models_ready else [],
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
            {"id": m, "object": "model", "created": now, "owned_by": "infinity-emb"}
            for m in embedding_service.list_models()
        ],
    }


@app.get("/stats")
async def stats():
    return {
        "uptime_seconds": int(time.time() - start_time),
        "models_ready":   models_ready,
        "models":         embedding_service.list_models() if models_ready else [],
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
    texts     = [req.input] if isinstance(req.input, str) else req.input
    model_id  = req.model or EMBED_MODEL
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    try:
        vecs, usage = await embedding_service.embed(model_id, texts)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not loaded. Available: {embedding_service.list_models()}")

    return EmbeddingResponse(
        model=model_id,
        data=[{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
        usage={"prompt_tokens": usage, "total_tokens": usage},
    )


# ---------------------------------------------------------------------------
# POST /v1/rerank
# ---------------------------------------------------------------------------
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    _check_models()
    request_stats["rerank"] += 1
    model_id = req.model or RERANK_MODEL
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    try:
        scores, usage = await embedding_service.rerank(model_id, req.query, req.documents)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not loaded. Available: {embedding_service.list_models()}")

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_n  = req.top_n or len(ranked)
    return RerankResponse(
        model=model_id,
        results=[
            {
                "index":    idx,
                "score":    float(s),
                "document": req.documents[idx] if req.return_documents else None,
            }
            for idx, s in ranked[:top_n]
        ],
        usage={"prompt_tokens": usage, "total_tokens": usage},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

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