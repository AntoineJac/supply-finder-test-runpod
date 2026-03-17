"""
Pydantic schemas for the GLM-4.7-Flash inference server.

Sampling defaults follow Z.ai + Unsloth official recommendations:
  General use:    temperature=1.0, top_p=0.95, min_p=0.01
  Tool-calling:   temperature=0.7, top_p=1.0,  min_p=0.01
  repeat_penalty: always 1.0 (disabled) — GLM loops with any other value
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------
class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    # FIX #6: Z.ai recommended defaults
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    min_p: float = Field(default=0.01, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    # NOTE: always forced to 1.0 server-side; exposed here for client clarity only
    repetition_penalty: float = Field(default=1.0, description="Must stay 1.0 for GLM-4.7 — looping otherwise")
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    # FIX #6: Z.ai recommended defaults
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    min_p: float = Field(default=0.01, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    # GLM-specific: False = direct answer (fast), True = chain-of-thought reasoning
    enable_thinking: bool = Field(
        default=False,
        description="False = direct answer (faster). True = chain-of-thought reasoning mode.",
    )


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


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


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None