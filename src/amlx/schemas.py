from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


class ThinkingConfig(BaseModel):
    enabled: bool = True
    budget_tokens: int | None = Field(default=None, ge=32, le=8192)
    summary: Literal["auto", "detailed"] = "auto"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    thinking: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False
    tools: list[ToolSpec] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceObject | None = "auto"
    thinking: ThinkingConfig | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    metadata: dict[str, Any] | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0


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


class ModelTrainRequest(BaseModel):
    model_id: str
    dataset_text: str | None = None
    samples: list[str] | None = None
    epochs: int = Field(default=1, ge=1, le=10)
    fine_tune_type: Literal["lora", "qlora", "dora"] = "qlora"
