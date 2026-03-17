"""
Pydantic request/response models for the GLM-4.7-Flash inference server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------
class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: List[CompletionChoice]
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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    # GLM-specific: set False to skip chain-of-thought (faster responses)
    enable_thinking: bool = Field(
        default=False,
        description="Enable GLM reasoning/thinking mode. False = faster direct answers.",
    )


class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: List[EmbeddingObject]
    usage: Dict[str, int]


# ---------------------------------------------------------------------------
# /v1/rerank
# ---------------------------------------------------------------------------
class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="List of documents to rank")
    top_n: Optional[int] = Field(None, description="Return top N results (default: all)")
    return_documents: bool = Field(
        default=False, description="Include document text in results"
    )


class RerankResult(BaseModel):
    index: int
    score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    model: str
    results: List[RerankResult]
    usage: Dict[str, int]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None