"""
RunPod Load Balancing Worker — Embeddings
Uses sentence-transformers with CUDA for GPU-accelerated inference.

Endpoints
---------
GET  /ping          → Health check (required by RunPod LB)
GET  /              → API info + readiness
GET  /v1/models     → List loaded models
POST /v1/embeddings → Sentence embeddings (OpenAI-compat)
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
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

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
EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
PORT        = int(os.getenv("PORT", 80))

# ---------------------------------------------------------------------------
# Models & state
# ---------------------------------------------------------------------------
embed_model:  SentenceTransformer | None = None
models_ready  = False
start_time    = time.time()
request_stats = {"embeddings": 0, "errors": 0}


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


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global embed_model, models_ready
    logger.info(f"Loading embed model: {EMBED_MODEL}")
    embed_model  = SentenceTransformer(EMBED_MODEL)
    models_ready = True
    logger.info("✓ Model loaded.")
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
        "models":  [EMBED_MODEL] if models_ready else [],
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": EMBED_MODEL, "object": "model", "created": now, "owned_by": "sentence-transformers"}
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)