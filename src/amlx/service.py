from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from amlx.adapters.base import ModelAdapter
from amlx.cache.prefix import PrefixCache
from amlx.scheduler import BatchScheduler
from amlx.schemas import (
    ChatChoice,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatMessage,
    Usage,
)


@dataclass(slots=True)
class InferenceService:
    adapter: ModelAdapter
    cache: PrefixCache
    scheduler: BatchScheduler

    @staticmethod
    def _render_prompt(messages: list[ChatMessage]) -> str:
        # Simple role-tag prompt; replace with chat template per model later.
        return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)

    def complete(self, req: ChatCompletionsRequest) -> ChatCompletionsResponse:
        prompt = self._render_prompt(req.messages)
        key = self.cache.build_key(req.model, prompt, req.temperature, req.max_tokens)
        cached = self.cache.get(key)

        if cached is not None:
            text = cached
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(text) // 4)
        else:
            result = self.scheduler.submit(
                model=req.model,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            text = result.text
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            self.cache.put(key, text)

        return ChatCompletionsResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def preload_model(self, model: str) -> bool:
        return self.adapter.preload_model(model)

    def unload_model(self, model: str) -> bool:
        return self.adapter.unload_model(model)
