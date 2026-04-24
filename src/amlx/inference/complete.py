from __future__ import annotations

import time
import uuid

from amlx.schemas import ChatChoice, ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage, Usage


class InferenceCompleteMixin:
    def complete(self, req: ChatCompletionsRequest) -> ChatCompletionsResponse:
        planned_tool_call = self._plan_tool_call(req)
        if planned_tool_call is not None:
            prompt = self._render_prompt(req.messages)
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(planned_tool_call.function.arguments) // 4)
            thinking, reasoning_tokens = self._build_thinking(req, planned_tool_call=planned_tool_call)
            return ChatCompletionsResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=req.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="", tool_calls=[planned_tool_call], thinking=thinking),
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

        prompt = self._render_prompt(req.messages)
        tools_prompt = self._render_tools_prompt(req.tools)
        if tools_prompt:
            prompt = f"{tools_prompt}\n{prompt}"
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
