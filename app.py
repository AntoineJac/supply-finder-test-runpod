    """
    RunPod NLP Inference Service
    Endpoints:
    POST /embed          — sentence embeddings (paraphrase-multilingual-mpnet-base-v2)
    POST /rerank         — cross-encoder reranking (ms-marco-MiniLM-L4-v2)
    POST /complete       — raw completion (GLM-4.7-Flash via SGLang/OpenAI-compat)
    POST /chat           — chat completion (GLM-4.7-Flash via SGLang/OpenAI-compat)
    """

    import os
    import time
    import logging
    from contextlib import asynccontextmanager
    from typing import List, Optional

    import torch
    import numpy as np
    from fastapi import FastAPI, HTTPException, Security
    from fastapi.security.api_key import APIKeyHeader
    from pydantic import BaseModel, Field
    import httpx

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # ── Config ────────────────────────────────────────────────────────────────────
    API_KEY          = os.environ.get("NLP_API_KEY", "")          # optional auth
    SGLANG_BASE_URL  = os.environ.get("SGLANG_BASE_URL", "http://localhost:30000")
    EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    CE_MODEL_NAME    = "cross-encoder/ms-marco-MiniLM-L4-v2"
    GLM_MODEL_NAME   = "glm-4.7-flash"  # served-model-name passed to SGLang

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {DEVICE}")

    # ── Auth ──────────────────────────────────────────────────────────────────────
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def verify_key(key: str = Security(api_key_header)):
        if API_KEY and key != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return key

    # ── Model singletons ──────────────────────────────────────────────────────────
    _embed_model  = None
    _cross_encoder = None

    def get_embed_model():
        global _embed_model
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model {EMBED_MODEL_NAME}...")
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
            # Warmup
            _embed_model.encode(["warmup"], show_progress_bar=False)
            logger.info("Embedding model ready.")
        return _embed_model

    def get_cross_encoder():
        global _cross_encoder
        if _cross_encoder is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"Loading cross-encoder {CE_MODEL_NAME}...")
            _cross_encoder = CrossEncoder(CE_MODEL_NAME, device=DEVICE, max_length=512)
            # Warmup — forces CUDA graph / kernel compilation
            _cross_encoder.predict([("warmup", "warmup")], show_progress_bar=False)
            logger.info("Cross-encoder ready.")
        return _cross_encoder

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Eagerly load both small models at startup so first request isn't slow
        get_embed_model()
        get_cross_encoder()
        # SGLang is a separate process — just verify it's up
        async with httpx.AsyncClient(timeout=10) as client:
            for attempt in range(12):
                try:
                    r = await client.get(f"{SGLANG_BASE_URL}/health")
                    if r.status_code == 200:
                        logger.info("SGLang server is healthy.")
                        break
                except Exception:
                    pass
                logger.info(f"Waiting for SGLang... attempt {attempt+1}/12")
                time.sleep(5)
            else:
                logger.warning("SGLang did not become healthy in time — /complete and /chat may fail.")
        yield

    app = FastAPI(title="NLP Inference Service", lifespan=lifespan)

    # ── Schemas ───────────────────────────────────────────────────────────────────

    class EmbedRequest(BaseModel):
        texts: List[str] = Field(..., min_length=1, max_length=512)
        batch_size: int = Field(64, ge=1, le=256)
        normalize: bool = False

    class EmbedResponse(BaseModel):
        embeddings: List[List[float]]
        model: str
        elapsed_ms: float

    class RerankRequest(BaseModel):
        query: str
        passages: List[str] = Field(..., min_length=1, max_length=2000)
        batch_size: int = Field(128, ge=1, le=512)
        return_raw: bool = True  # return raw logits (needed for your -6 gate)

    class RerankResult(BaseModel):
        index: int
        score: float

    class RerankResponse(BaseModel):
        results: List[RerankResult]   # sorted by score desc
        model: str
        elapsed_ms: float

    class Message(BaseModel):
        role: str
        content: str

    class CompleteRequest(BaseModel):
        prompt: str
        max_tokens: int = Field(1024, ge=1, le=8192)
        temperature: float = Field(0.0, ge=0.0, le=2.0)
        top_p: float = Field(1.0, ge=0.0, le=1.0)
        stop: Optional[List[str]] = None
        stream: bool = False

    class ChatRequest(BaseModel):
        messages: List[Message]
        max_tokens: int = Field(1024, ge=1, le=8192)
        temperature: float = Field(0.0, ge=0.0, le=2.0)
        top_p: float = Field(1.0, ge=0.0, le=1.0)
        stop: Optional[List[str]] = None
        stream: bool = False
        response_format: Optional[dict] = None  # e.g. {"type": "json_object"}

    # ── /embed ────────────────────────────────────────────────────────────────────

    @app.post("/embed", response_model=EmbedResponse, dependencies=[Security(verify_key)])
    async def embed(req: EmbedRequest):
        t0 = time.perf_counter()
        model = get_embed_model()

        embeddings = model.encode(
            req.texts,
            batch_size=req.batch_size,
            show_progress_bar=False,
            normalize_embeddings=req.normalize,
            convert_to_numpy=True,
        )

        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=EMBED_MODEL_NAME,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ── /rerank ───────────────────────────────────────────────────────────────────

    @app.post("/rerank", response_model=RerankResponse, dependencies=[Security(verify_key)])
    async def rerank(req: RerankRequest):
        t0 = time.perf_counter()
        model = get_cross_encoder()

        if not req.passages:
            return RerankResponse(results=[], model=CE_MODEL_NAME, elapsed_ms=0)

        pairs = [(req.query, p) for p in req.passages]

        # batch_size tuning: GPU can handle 128+, CPU should stay at 64
        effective_batch = req.batch_size if DEVICE == "cuda" else min(req.batch_size, 64)

        raw_scores: np.ndarray = model.predict(
            pairs,
            batch_size=effective_batch,
            show_progress_bar=False,
        )

        results = [
            RerankResult(index=i, score=float(s))
            for i, s in enumerate(raw_scores)
        ]
        results.sort(key=lambda x: x.score, reverse=True)

        return RerankResponse(
            results=results,
            model=CE_MODEL_NAME,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ── /complete ─────────────────────────────────────────────────────────────────

    @app.post("/complete", dependencies=[Security(verify_key)])
    async def complete(req: CompleteRequest):
        """Raw text completion — wraps SGLang's OpenAI-compatible /v1/completions."""
        payload = {
            "model": GLM_MODEL_NAME,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": req.stream,
        }
        if req.stop:
            payload["stop"] = req.stop

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                r = await client.post(f"{SGLANG_BASE_URL}/v1/completions", json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"SGLang error: {e}")

    # ── /chat ─────────────────────────────────────────────────────────────────────

    @app.post("/chat", dependencies=[Security(verify_key)])
    async def chat(req: ChatRequest):
        """Chat completion — wraps SGLang's OpenAI-compatible /v1/chat/completions."""
        payload = {
            "model": GLM_MODEL_NAME,
            "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": req.stream,
        }
        if req.stop:
            payload["stop"] = req.stop
        if req.response_format:
            payload["response_format"] = req.response_format

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                r = await client.post(f"{SGLANG_BASE_URL}/v1/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"SGLang error: {e}")

    # ── Health ────────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "device": DEVICE}