from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False
    metadata: dict[str, Any] | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class ModelDownloadRequest(BaseModel):
    model_id: str


class RuntimePowerRequest(BaseModel):
    gpu_limit_percent: int | None = Field(default=None, ge=20, le=100)
    max_model_b: float | None = Field(default=None, ge=0.0, le=200.0)
