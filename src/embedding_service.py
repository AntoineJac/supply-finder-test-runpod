"""
Embedding & Reranking service backed by infinity-emb AsyncEngineArray.

Models are configured via environment variables:
  MODEL_NAMES  — semicolon-separated HuggingFace model IDs (required)
  BATCH_SIZES  — semicolon-separated ints    (default: 32 each)
  DTYPES       — semicolon-separated strings (default: auto each)

Example:
  MODEL_NAMES=sentence-transformers/paraphrase-multilingual-mpnet-base-v2;cross-encoder/ms-marco-MiniLM-L4-v2
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import cached_property

from infinity_emb.engine import AsyncEngineArray, EngineArgs

logger = logging.getLogger("nlp-worker")

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_DTYPE      = "auto"
_DEFAULT_BACKEND    = "torch"

# Tune inference queue depth for GPU throughput
if not os.environ.get("INFINITY_QUEUE_SIZE"):
    os.environ["INFINITY_QUEUE_SIZE"] = "48000"


class EmbeddingServiceConfig:
    @cached_property
    def model_names(self) -> list[str]:
        raw = os.environ.get("MODEL_NAMES", "")
        if not raw:
            raise ValueError(
                "Missing required env var MODEL_NAMES.\n"
                "Provide one or more HuggingFace model IDs separated by semicolons.\n"
                "  Example: MODEL_NAMES=sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                ";cross-encoder/ms-marco-MiniLM-L4-v2"
            )
        return [m for m in raw.split(";") if m]

    def _multi(self, name: str, default) -> list[str]:
        raw = os.getenv(name)
        if raw is None:
            return [str(default)] * len(self.model_names)
        vals = [v.strip() for v in raw.split(";") if v.strip()]
        if len(vals) != len(self.model_names):
            raise ValueError(
                f"Env var {name} must have the same number of elements as MODEL_NAMES"
            )
        return vals

    @cached_property
    def batch_sizes(self) -> list[int]:
        return [int(v) for v in self._multi("BATCH_SIZES", _DEFAULT_BATCH_SIZE)]

    @cached_property
    def dtypes(self) -> list[str]:
        return self._multi("DTYPES", _DEFAULT_DTYPE)


class EmbeddingService:
    def __init__(self):
        self.config     = EmbeddingServiceConfig()
        self._semaphore: asyncio.Semaphore | None = None
        self.is_running = False

        engine_args = [
            EngineArgs(
                model_name_or_path=name,
                batch_size=bs,
                engine=_DEFAULT_BACKEND,
                dtype=dtype,
                bettertransformer=False,  # optimum.bettertransformer removed in optimum>=1.23
                model_warmup=False,
                lengths_via_tokenize=True,
            )
            for name, bs, dtype in zip(
                self.config.model_names,
                self.config.batch_sizes,
                self.config.dtypes,
            )
        ]
        self.engine_array = AsyncEngineArray.from_args(engine_args)

    async def start(self) -> None:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)
        async with self._semaphore:
            if not self.is_running:
                logger.info(f"Starting engines for: {self.config.model_names}")
                await self.engine_array.astart()
                self.is_running = True
                logger.info("✓ All engines started.")

    async def stop(self) -> None:
        if self._semaphore is None:
            return
        async with self._semaphore:
            if self.is_running:
                await self.engine_array.astop()
                self.is_running = False

    def list_models(self) -> list[str]:
        return list(self.engine_array.engines_dict.keys())

    async def embed(self, model: str, texts: list[str]):
        """Returns (embeddings: list[ndarray], usage: int)."""
        if not self.is_running:
            await self.start()
        return await self.engine_array[model].embed(texts)

    async def rerank(self, model: str, query: str, docs: list[str]):
        """Returns (scores: list[float], usage: int)."""
        if not self.is_running:
            await self.start()
        return await self.engine_array[model].rerank(
            query=query, docs=docs, raw_scores=False
        )