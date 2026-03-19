"""
RunPod Load Balancing Worker — Embeddings + Reranker
Uses sentence-transformers with CUDA for GPU-accelerated inference.

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

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

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

# ---------------------------------------------------------------------------
# Models & state
# ---------------------------------------------------------------------------
embed_model:  SentenceTransformer | None = None
rerank_model: CrossEncoder | None        = None
models_ready  = False
start_time    = time.time()
request_stats = {"embeddings": 0, "rerank": 0, "errors": 0}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Dict[str, int]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: bool = False
    raw_scores: bool = Field(default=True, description="Return raw logit scores. Set false for sigmoid-normalised 0–1 scores")


class RerankResponse(BaseModel):
    model: str
    results: List[Dict[str, Any]]
    usage: Dict[str, int]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global embed_model, rerank_model, models_ready
    logger.info(f"Loading embed model: {EMBED_MODEL}")
    embed_model  = SentenceTransformer(EMBED_MODEL)
    logger.info(f"Loading rerank model: {RERANK_MODEL}")
    rerank_model = CrossEncoder(RERANK_MODEL)
    models_ready = True
    logger.info("✓ All models loaded.")
    yield
    models_ready = False
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="NLP Inference Worker", version="1.0.0", lifespan=lifespan)


def _check_ready():
    if not models_ready:
        request_stats["errors"] += 1
        raise HTTPException(status_code=503, detail="Models not ready — check /ping")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/ping")
async def health_check():
    if models_ready:
        return {"status": "healthy"}
    return JSONResponse(
        content={"status": "initializing"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/")
async def root():
    return {
        "service": "NLP Inference Worker",
        "ready":   models_ready,
        "models":  [EMBED_MODEL, RERANK_MODEL] if models_ready else [],
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": now, "owned_by": "sentence-transformers"}
            for m in [EMBED_MODEL, RERANK_MODEL]
        ],
    }


@app.get("/stats")
async def stats():
    return {
        "uptime_seconds": int(time.time() - start_time),
        "models_ready":   models_ready,
        "request_counts": request_stats,
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(req: EmbeddingRequest):
    _check_ready()
    request_stats["embeddings"] += 1
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    try:
        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(None, embed_model.encode, texts)
    except Exception as exc:
        request_stats["errors"] += 1
        logger.exception("embed failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return EmbeddingResponse(
        model=EMBED_MODEL,
        data=[{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
        usage={"prompt_tokens": len(texts), "total_tokens": len(texts)},
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    _check_ready()
    request_stats["rerank"] += 1
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")
    try:
        pairs  = [[req.query, doc] for doc in req.documents]
        loop   = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, rerank_model.predict, pairs)
    except Exception as exc:
        request_stats["errors"] += 1
        logger.exception("rerank failed")
        raise HTTPException(status_code=500, detail=str(exc))
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_n   = req.top_n or len(indexed)
    return RerankResponse(
        model=RERANK_MODEL,
        results=[
            {
                "index":    i,
                "score":    float(s),
                "document": req.documents[i] if req.return_documents else None,
            }
            for i, s in indexed[:top_n]
        ],
        usage={"prompt_tokens": len(req.documents), "total_tokens": len(req.documents)},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)