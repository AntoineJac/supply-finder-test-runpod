"""
RunPod Load Balancing Worker — GLM-4.7-Flash + Embeddings + Reranker
Follows: https://docs.runpod.io/serverless/load-balancing/vllm-worker

Endpoints
---------
GET  /ping                      → Health check (required by RunPod LB)
GET  /                          → API info + readiness
GET  /v1/models                 → List available models (OpenAI-compat)
POST /v1/completions            → Raw text completion  (OpenAI-compat, streaming ✓)
POST /v1/chat/completions       → Chat completion      (OpenAI-compat, streaming ✓)
POST /v1/embeddings             → Sentence embeddings  (OpenAI-compat)
POST /v1/rerank                 → Cross-encoder rerank score
GET  /stats                     → Live server stats
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from sentence_transformers import CrossEncoder, SentenceTransformer
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
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
logger = logging.getLogger("glm47-worker")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
MODEL_NAME        = os.getenv("MODEL_NAME",        "zai-org/GLM-4.7-Flash")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "glm-4.7-flash")
EMBED_MODEL_NAME  = os.getenv("EMBED_MODEL_NAME",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L4-v2")
TP_SIZE           = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEM_UTIL      = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.72"))  # leaves room for aux models
MAX_MODEL_LEN     = int(os.getenv("MAX_MODEL_LEN", "32768"))
PORT              = int(os.getenv("PORT", 80))
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
engine:        Optional[AsyncLLMEngine]    = None
embed_model:   Optional[SentenceTransformer] = None
rerank_model:  Optional[CrossEncoder]      = None
engine_ready:  bool = False
aux_ready:     bool = False
start_time:    float = time.time()
request_stats: dict = {"completions": 0, "chat": 0, "embeddings": 0, "rerank": 0, "errors": 0}


# ---------------------------------------------------------------------------
# Lifespan — load everything on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, embed_model, rerank_model, engine_ready, aux_ready

    # ── Aux models first (small, fast) ──────────────────────────────────────
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

    logger.info(f"Loading reranker model: {RERANK_MODEL_NAME}")
    rerank_model = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
    aux_ready = True
    logger.info("Aux models ready.")

    # ── vLLM engine ─────────────────────────────────────────────────────────
    logger.info(f"Initialising vLLM engine: {MODEL_NAME}")
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        served_model_name=SERVED_MODEL_NAME,
        tensor_parallel_size=TP_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=16384,
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        trust_remote_code=True,
        tool_call_parser="glm47",
        reasoning_parser="glm45",
        enable_auto_tool_choice=True,
        disable_log_requests=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_ready = True
    logger.info("vLLM engine ready. All systems go.")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down.")
    engine = None
    engine_ready = False
    aux_ready = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GLM-4.7-Flash Inference Server",
    version="1.0.0",
    description="RunPod Load Balancer worker — GLM-4.7-Flash + embeddings + reranker",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check  ← REQUIRED by RunPod load balancer
# ---------------------------------------------------------------------------
@app.get("/ping")
async def health_check():
    """
    RunPod load balancer calls this endpoint to decide if the worker is ready.
    • Return HTTP 200  → worker is healthy and receives traffic
    • Return HTTP 204  → worker is initialising, keep waiting
    """
    if engine_ready and aux_ready:
        return {"status": "healthy"}
    return JSONResponse(
        content={"status": "initializing"},
        status_code=status.HTTP_204_NO_CONTENT,
    )


# ---------------------------------------------------------------------------
# Root / info
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "GLM-4.7-Flash Inference Worker",
        "ready": engine_ready and aux_ready,
        "models": {
            "llm": SERVED_MODEL_NAME,
            "embeddings": EMBED_MODEL_NAME,
            "reranker": RERANK_MODEL_NAME,
        },
        "endpoints": {
            "health":       "GET  /ping",
            "models":       "GET  /v1/models",
            "completion":   "POST /v1/completions",
            "chat":         "POST /v1/chat/completions",
            "embeddings":   "POST /v1/embeddings",
            "rerank":       "POST /v1/rerank",
            "stats":        "GET  /stats",
        },
    }


# ---------------------------------------------------------------------------
# Models list  (OpenAI-compat)
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": SERVED_MODEL_NAME,  "object": "model", "created": now, "owned_by": "zai-org"},
            {"id": EMBED_MODEL_NAME,   "object": "model", "created": now, "owned_by": "sentence-transformers"},
            {"id": RERANK_MODEL_NAME,  "object": "model", "created": now, "owned_by": "cross-encoder"},
        ],
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@app.get("/stats")
async def stats():
    uptime = int(time.time() - start_time)
    return {
        "uptime_seconds": uptime,
        "engine_ready": engine_ready,
        "aux_ready": aux_ready,
        "device": DEVICE,
        "request_counts": request_stats,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_engine():
    if not engine_ready or engine is None:
        request_stats["errors"] += 1
        raise HTTPException(status_code=503, detail="LLM engine not ready")

def _check_aux():
    if not aux_ready or embed_model is None or rerank_model is None:
        request_stats["errors"] += 1
        raise HTTPException(status_code=503, detail="Aux models not ready")

def _sampling_params(req) -> SamplingParams:
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=getattr(req, "top_k", -1),
        frequency_penalty=getattr(req, "frequency_penalty", 0.0),
        presence_penalty=getattr(req, "presence_penalty", 0.0),
        stop=req.stop if req.stop else None,
    )


# ---------------------------------------------------------------------------
# POST /v1/completions  — raw text completion (OpenAI-compat)
# ---------------------------------------------------------------------------
@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    _check_engine()
    request_stats["completions"] += 1
    sampling = _sampling_params(req)
    request_id = random_uuid()

    if req.stream:
        return StreamingResponse(
            _stream_completion(req.prompt, sampling, request_id, req.model or SERVED_MODEL_NAME),
            media_type="text/event-stream",
        )

    final = None
    async for out in engine.generate(req.prompt, sampling, request_id):
        final = out

    if final is None:
        raise HTTPException(status_code=500, detail="No output generated")

    choice = final.outputs[0]
    prompt_tokens = len(final.prompt_token_ids) if final.prompt_token_ids else 0
    completion_tokens = len(choice.token_ids)

    return CompletionResponse(
        id=f"cmpl-{request_id}",
        model=req.model or SERVED_MODEL_NAME,
        choices=[{
            "text": choice.text,
            "index": 0,
            "finish_reason": choice.finish_reason,
            "logprobs": None,
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


async def _stream_completion(
    prompt: str, sampling: SamplingParams, request_id: str, model: str
) -> AsyncGenerator[str, None]:
    try:
        async for out in engine.generate(prompt, sampling, request_id):
            for choice in out.outputs:
                chunk = {
                    "id": f"cmpl-{request_id}",
                    "object": "text_completion",
                    "model": model,
                    "choices": [{
                        "text": choice.text,
                        "index": 0,
                        "finish_reason": choice.finish_reason,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions  — chat (OpenAI-compat, streaming ✓)
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    _check_engine()
    request_stats["chat"] += 1

    # Build prompt via the model's own chat template (vLLM handles this natively)
    # We pass messages directly to vLLM which applies GLM's template
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest as VllmChatReq,
    )

    request_id = random_uuid()
    sampling = _sampling_params(req)

    # Convert messages to prompt string using GLM's tokenizer chat template
    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    # Use vLLM's tokenizer to apply the model's own chat template
    tokenizer = await engine.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages_dicts,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": req.enable_thinking},
    )

    if req.stream:
        return StreamingResponse(
            _stream_chat(prompt, sampling, request_id, req.model or SERVED_MODEL_NAME),
            media_type="text/event-stream",
        )

    final = None
    async for out in engine.generate(prompt, sampling, request_id):
        final = out

    if final is None:
        raise HTTPException(status_code=500, detail="No output generated")

    choice = final.outputs[0]
    prompt_tokens = len(final.prompt_token_ids) if final.prompt_token_ids else 0
    completion_tokens = len(choice.token_ids)

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        model=req.model or SERVED_MODEL_NAME,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": choice.text},
            "finish_reason": choice.finish_reason,
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


async def _stream_chat(
    prompt: str, sampling: SamplingParams, request_id: str, model: str
) -> AsyncGenerator[str, None]:
    try:
        async for out in engine.generate(prompt, sampling, request_id):
            for choice in out.outputs:
                chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": choice.text},
                        "finish_reason": choice.finish_reason,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ---------------------------------------------------------------------------
# POST /v1/embeddings  — sentence embeddings (OpenAI-compat)
# ---------------------------------------------------------------------------
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(req: EmbeddingRequest):
    _check_aux()
    request_stats["embeddings"] += 1

    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    loop = asyncio.get_event_loop()
    vecs: np.ndarray = await loop.run_in_executor(
        None,
        lambda: embed_model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
    )

    total_tokens = sum(len(t.split()) for t in texts)
    return EmbeddingResponse(
        model=req.model or EMBED_MODEL_NAME,
        data=[
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(vecs)
        ],
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


# ---------------------------------------------------------------------------
# POST /v1/rerank  — cross-encoder reranking
# ---------------------------------------------------------------------------
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    _check_aux()
    request_stats["rerank"] += 1

    if not req.documents:
        raise HTTPException(status_code=400, detail="documents must not be empty")

    pairs = [[req.query, doc] for doc in req.documents]

    loop = asyncio.get_event_loop()
    scores: np.ndarray = await loop.run_in_executor(
        None,
        lambda: rerank_model.predict(pairs, convert_to_numpy=True),
    )

    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
    top_n = req.top_n or len(ranked)

    results = [
        {
            "index": idx,
            "score": float(score),
            "document": req.documents[idx] if req.return_documents else None,
        }
        for idx, score in ranked[:top_n]
    ]

    return RerankResponse(
        model=req.model or RERANK_MODEL_NAME,
        results=results,
        usage={"prompt_tokens": sum(len(d.split()) for d in req.documents)},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")