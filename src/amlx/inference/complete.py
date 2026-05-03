from __future__ import annotations

import time
import uuid

from amlx.schemas import ChatChoice, ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage, Usage


class InferenceCompleteMixin:
    def complete(self, req: ChatCompletionsRequest) -> ChatCompletionsResponse:
        prompt = self._build_prompt(req)
        has_tools = bool(req.tools) and req.tool_choice != "none"

        # Skip cache when tools are active so every call gets a fresh decision
        key = self.cache.build_key(req.model, prompt, req.temperature, req.max_tokens)
        _cached = None if req.no_cache or has_tools else self.cache.get(key)
        cached = _cached if _cached else None

        if cached is not None:
            raw_text = cached
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(raw_text) // 4)
        else:
            result = self.scheduler.submit(
                model=req.model,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            raw_text = result.text
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            if not req.no_cache and not has_tools:
                self.cache.put(key, raw_text)

        tool_call = self._parse_tool_call(raw_text) if has_tools else None

        if tool_call is not None:
            thinking, reasoning_tokens = self._build_thinking(req, planned_tool_call=tool_call)
            return ChatCompletionsResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=req.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="", tool_calls=[tool_call], thinking=thinking),
                        finish_reason="tool_calls",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                ),
            )

        text = self._strip_thinking(raw_text)
        thinking, reasoning_tokens = self._build_thinking(req, planned_tool_call=None, generated_text=text)
        return ChatCompletionsResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text, thinking=thinking),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        )

    def preload_model(self, model: str) -> bool:
        return self.adapter.preload_model(model)

    def unload_model(self, model: str) -> bool:
        return self.adapter.unload_model(model)
