"""
Pydantic schemas for the NLP inference worker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /v1/embeddings  (OpenAI-compat)
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Dict[str, int]


# ---------------------------------------------------------------------------
# /v1/rerank
# ---------------------------------------------------------------------------
class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="Documents to rank against the query")
    top_n: Optional[int] = Field(None, description="Return top N results (default: all)")
    return_documents: bool = Field(default=False, description="Include document text in results")


class RerankResponse(BaseModel):
    model: str
    results: List[Dict[str, Any]]
    usage: Dict[str, int]