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

Fixes vs previous version
--------------------------
FIX #1 — AsyncEngineArgs: removed CLI-only flags that are NOT valid Python kwargs:
         tool_call_parser, reasoning_parser, enable_auto_tool_choice,
         disable_log_requests → all cause TypeError on startup.

FIX #2 — OOM on 24 GB: switched to unsloth/GLM-4.7-Flash-FP8-Dynamic.
         zai-org BF16 weights = ~30-32 GB; FP8-Dynamic = ~16 GB.
         Also lowered MAX_MODEL_LEN default to 8192 — the KV cache
         pre-allocation for 32768 tokens alone can OOM a 4090.

FIX #3 — Removed MTP speculative decoding. On Ampere (RTX 4090) MTP causes
         10x throughput regression per Unsloth. Only safe on Hopper/B200.

FIX #4 — AsyncLLMEngine.from_engine_args() is synchronous and blocks for
         2-3 min. Runs in executor so the event loop stays unblocked and
         /ping can still respond with 204 during load.

FIX #5 — chat_completions: removed stray unused import. Fixed
         apply_chat_template: enable_thinking passed as direct kwarg
         (GLM template reads it), with TypeError fallback for older
         transformers builds.

FIX #6 — Default sampling: temperature=1.0, top_p=0.95, min_p=0.01
         per Z.ai + Unsloth official recommendations.
         repetition_penalty=1.0 (disabled) — critical for GLM, looping
         occurs with any other value.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

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
# Config — all overridable via RunPod env vars
# ---------------------------------------------------------------------------
# FIX #2: FP8-Dynamic fits 24 GB; BF16 original needs ~30+ GB
MODEL_NAME        = os.getenv("MODEL_NAME",        "unsloth/GLM-4.7-Flash-FP8-Dynamic")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "glm-4.7-flash")
EMBED_MODEL_NAME  = os.getenv("EMBED_MODEL_NAME",  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L4-v2")
TP_SIZE           = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
# FP8 weights ~16 GB + aux ~1.1 GB → 0.85 util leaves ~3.5 GB for KV cache on 4090
GPU_MEM_UTIL      = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
# FIX #2: 8192 safe on 24 GB; increase via env var on larger GPUs
MAX_MODEL_LEN     = int(os.getenv("MAX_MODEL_LEN", "8192"))
PORT              = int(os.getenv("PORT", 80))
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
engine:        Optional[AsyncLLMEngine]     = None
embed_model:   Optional[SentenceTransformer] = None
rerank_model:  Optional[CrossEncoder]       = None
engine_ready:  bool  = False
aux_ready:     bool  = False
start_time:    float = time.time()
request_stats: dict  = {"completions": 0, "chat": 0, "embeddings": 0, "rerank": 0, "errors": 0}


# ---------------------------------------------------------------------------
# Lifespan — aux models first, then vLLM
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, embed_model, rerank_model, engine_ready, aux_ready
    loop = asyncio.get_event_loop()

    # ── Aux models (small, loads in ~10 s) ──────────────────────────────────
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = await loop.run_in_executor(
        None, lambda: SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    )
    logger.info(f"Loading reranker model: {RERANK_MODEL_NAME}")
    rerank_model = await loop.run_in_executor(
        None, lambda: CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
    )
    aux_ready = True
    logger.info("✓ Aux models ready.")

    # ── vLLM engine (~2-3 min to load) ──────────────────────────────────────
    logger.info(f"Initialising vLLM engine: {MODEL_NAME}")

    # FIX #1: Only pass valid AsyncEngineArgs kwargs.
    # CLI-only flags (tool_call_parser, reasoning_parser, enable_auto_tool_choice,
    # disable_log_requests) are NOT valid here and cause TypeError.
    # FIX #3: No MTP speculative decoding — 10x slower on Ampere (RTX 4090).
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
        seed=3407,
    )

    # FIX #4: from_engine_args() is sync and blocks for minutes — run in executor
    engine = await loop.run_in_executor(
        None, lambda: AsyncLLMEngine.from_engine_args(engine_args)
    )
    engine_ready = True
    logger.info("✓ vLLM engine ready.")

    yield

    logger.info("Shutting down.")
    engine       = None
    engine_ready = False
    aux_ready    = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GLM-4.7-Flash Inference Server",
    version="2.0.0",
    description="RunPod Load Balancer worker — GLM-4.7-Flash FP8 + embeddings + reranker",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# /ping — REQUIRED by RunPod load balancer
# 200 = healthy (gets traffic)  |  204 = still initialising (wait)
# ---------------------------------------------------------------------------
@app.get("/ping")
async def health_check():
    if engine_ready and aux_ready:
        return {"status": "healthy"}
    return JSONResponse(
        content={"status": "initializing"},
        status_code=status.HTTP_204_NO_CONTENT,
    )


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
            "health":     "GET  /ping",
            "models":     "GET  /v1/models",
            "completion": "POST /v1/completions",
            "chat":       "POST /v1/chat/completions",
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
            {"id": SERVED_MODEL_NAME,  "object": "model", "created": now, "owned_by": "zai-org"},
            {"id": EMBED_MODEL_NAME,   "object": "model", "created": now, "owned_by": "sentence-transformers"},
            {"id": RERANK_MODEL_NAME,  "object": "model", "created": now, "owned_by": "cross-encoder"},
        ],
    }


@app.get("/stats")
async def stats():
    return {
        "uptime_seconds": int(time.time() - start_time),
        "engine_ready": engine_ready,
        "aux_ready": aux_ready,
        "device": DEVICE,
        "model": MODEL_NAME,
        "max_model_len": MAX_MODEL_LEN,
        "request_counts": request_stats,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_engine():
    if not engine_ready or engine is None:
        request_stats["errors"] += 1
        raise HTTPException(status_code=503, detail="LLM engine not ready — check /ping")


def _check_aux():
    if not aux_ready:
        request_stats["errors"] += 1
        raise HTTPException(status_code=503, detail="Aux models not ready")


def _build_sampling(req) -> SamplingParams:
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=getattr(req, "top_k", -1),
        min_p=getattr(req, "min_p", 0.01),           # Unsloth: set min_p=0.01 for llama.cpp parity
        frequency_penalty=getattr(req, "frequency_penalty", 0.0),
        presence_penalty=getattr(req, "presence_penalty", 0.0),
        # FIX #6: repetition_penalty MUST be 1.0 (disabled) for GLM-4.7
        # Any other value causes looping / poor output quality
        repetition_penalty=1.0,
        stop=req.stop or None,
    )


# ---------------------------------------------------------------------------
# POST /v1/completions
# ---------------------------------------------------------------------------
@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    _check_engine()
    request_stats["completions"] += 1
    prompt   = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    sampling = _build_sampling(req)
    req_id   = random_uuid()

    if req.stream:
        return StreamingResponse(
            _stream_completion(prompt, sampling, req_id, req.model or SERVED_MODEL_NAME),
            media_type="text/event-stream",
        )

    final = None
    async for out in engine.generate(prompt, sampling, req_id):
        final = out
    if final is None:
        raise HTTPException(status_code=500, detail="No output generated")

    choice = final.outputs[0]
    pt = len(final.prompt_token_ids) if final.prompt_token_ids else 0
    ct = len(choice.token_ids)
    return CompletionResponse(
        id=f"cmpl-{req_id}",
        model=req.model or SERVED_MODEL_NAME,
        choices=[{"text": choice.text, "index": 0, "finish_reason": choice.finish_reason, "logprobs": None}],
        usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    )


async def _stream_completion(
    prompt: str, sampling: SamplingParams, req_id: str, model: str
) -> AsyncGenerator[str, None]:
    try:
        async for out in engine.generate(prompt, sampling, req_id):
            for c in out.outputs:
                yield f"data: {json.dumps({'id': f'cmpl-{req_id}', 'object': 'text_completion', 'model': model, 'choices': [{'text': c.text, 'index': 0, 'finish_reason': c.finish_reason}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    _check_engine()
    request_stats["chat"] += 1
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    req_id   = random_uuid()
    sampling = _build_sampling(req)

    # FIX #5: apply GLM's chat template cleanly
    tokenizer      = await engine.get_tokenizer()
    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        # GLM-4.7 template accepts enable_thinking as a direct kwarg
        prompt = tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=req.enable_thinking,
        )
    except TypeError:
        # Fallback for transformers versions that don't forward extra kwargs
        prompt = tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )

    if req.stream:
        return StreamingResponse(
            _stream_chat(prompt, sampling, req_id, req.model or SERVED_MODEL_NAME),
            media_type="text/event-stream",
        )

    final = None
    async for out in engine.generate(prompt, sampling, req_id):
        final = out
    if final is None:
        raise HTTPException(status_code=500, detail="No output generated")

    choice = final.outputs[0]
    pt = len(final.prompt_token_ids) if final.prompt_token_ids else 0
    ct = len(choice.token_ids)
    return ChatCompletionResponse(
        id=f"chatcmpl-{req_id}",
        model=req.model or SERVED_MODEL_NAME,
        choices=[{"index": 0, "message": {"role": "assistant", "content": choice.text}, "finish_reason": choice.finish_reason}],
        usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    )


async def _stream_chat(
    prompt: str, sampling: SamplingParams, req_id: str, model: str
) -> AsyncGenerator[str, None]:
    try:
        async for out in engine.generate(prompt, sampling, req_id):
            for c in out.outputs:
                yield f"data: {json.dumps({'id': f'chatcmpl-{req_id}', 'object': 'chat.completion.chunk', 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': c.text}, 'finish_reason': c.finish_reason}]})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ---------------------------------------------------------------------------
# POST /v1/embeddings
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
        lambda: embed_model.encode(texts, batch_size=64, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False),
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
    _check_aux()
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