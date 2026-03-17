"""
core/clients/nlp_client.py
Thin client for the RunPod NLP inference service.
Drop-in replacement for the old local sentence-transformers client.

Environment variables:
  NLP_SERVICE_URL   — base URL of the RunPod service  (required)
  NLP_API_KEY       — API key sent as X-API-Key header (optional)
  NLP_TIMEOUT       — request timeout in seconds       (default: 120)
"""

import os
import logging
from typing import List

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = os.environ.get("NLP_SERVICE_URL", "").rstrip("/")
_API_KEY  = os.environ.get("NLP_API_KEY", "")
_TIMEOUT  = float(os.environ.get("NLP_TIMEOUT", "120"))

if not _BASE_URL:
    logger.warning("NLP_SERVICE_URL is not set — nlp_client calls will fail")

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _API_KEY:
        h["X-API-Key"] = _API_KEY
    return h

def _post(path: str, payload: dict) -> dict:
    url = f"{_BASE_URL}{path}"
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.post(url, json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()


# ── Embeddings ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[float]:
    """Encode a single text. Returns a list of floats (768-dim)."""
    data = _post("/embed", {"texts": [text], "normalize": False})
    return data["embeddings"][0]


def tokenize_batch(
    texts: List[str],
    batch_size: int = 64,
) -> List[List[float]]:
    """Batch encode texts. Returns list of embeddings."""
    data = _post("/embed", {"texts": texts, "batch_size": batch_size, "normalize": False})
    return data["embeddings"]


# ── Cross-encoder reranking ───────────────────────────────────────────────────

def cross_encode_batch(
    query: str,
    passages: List[str],
    batch_size: int = 128,
) -> List[float]:
    """
    Score (query, passage) pairs. Returns raw logit scores in INPUT order
    (not sorted) — same contract as the old local cross_encode_batch.
    """
    if not passages:
        return []

    data = _post("/rerank", {
        "query": query,
        "passages": passages,
        "batch_size": batch_size,
        "return_raw": True,
    })

    # Service returns results sorted by score — restore original index order
    results = data["results"]           # [{"index": int, "score": float}, ...]
    scores = [0.0] * len(passages)
    for r in results:
        scores[r["index"]] = r["score"]
    return scores


# ── LLM — completion ──────────────────────────────────────────────────────────

def complete(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: List[str] = None,
) -> str:
    """Raw text completion. Returns the generated text string."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    data = _post("/complete", payload)
    return data["choices"][0]["text"]


# ── LLM — chat ────────────────────────────────────────────────────────────────

def chat(
    messages: List[dict],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: List[str] = None,
    response_format: dict = None,
) -> str:
    """
    Chat completion. messages = [{"role": "user"|"system"|"assistant", "content": str}]
    Returns the assistant reply string.
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if stop:
        payload["stop"] = stop
    if response_format:
        payload["response_format"] = response_format
    data = _post("/chat", payload)
    return data["choices"][0]["message"]["content"]