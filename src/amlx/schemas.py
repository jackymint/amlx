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
    profile: str | None = None
    dataset_text: str | None = None
    samples: list[str] | None = None
    epochs: int = Field(default=1, ge=1, le=20)
    fine_tune_type: Literal["lora", "qlora", "dora"] = "qlora"
    learning_rate: float = Field(default=1e-5, gt=0)
    lora_rank: int = Field(default=8, ge=4, le=64)
    lora_layers: int = Field(default=16, ge=1, le=64)
    max_seq_length: int = Field(default=2048, ge=128, le=8192)


class ModelQuantizeRequest(BaseModel):
    model_id: str
    output_path: str = Field(min_length=1)
    q_bits: int = Field(default=4, ge=2, le=8)
    q_group_size: int = Field(default=64, ge=16, le=256)


class ModelTrainSaveRequest(BaseModel):
    task_id: str | None = None
    profile: str | None = None
    adapter_path: str | None = None
    effective_model: str | None = None
    output_path: str = Field(min_length=1)
