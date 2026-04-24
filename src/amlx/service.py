from __future__ import annotations

import json
import re
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
    ToolCall,
    ToolCallFunction,
    ToolChoiceObject,
    ToolSpec,
    Usage,
)


@dataclass(slots=True)
class InferenceService:
    adapter: ModelAdapter
    cache: PrefixCache
    scheduler: BatchScheduler

    @staticmethod
    def _message_to_prompt_line(message: ChatMessage) -> str:
        role = message.role.upper()
        content = (message.content or "").strip()
        if message.role == "tool":
            tool_name = message.name or "tool"
            call_id = f"[{message.tool_call_id}]" if message.tool_call_id else ""
            return f"TOOL {tool_name}{call_id}: {content}"
        if message.role == "assistant" and message.tool_calls:
            calls = "; ".join(f"{c.function.name}({c.function.arguments})" for c in message.tool_calls)
            return f"ASSISTANT TOOL_CALLS: {calls}"
        return f"{role}: {content}"

    def _render_prompt(self, messages: list[ChatMessage]) -> str:
        # Simple role-tag prompt; replace with chat template per model later.
        return "\n".join(self._message_to_prompt_line(m) for m in messages)

    @staticmethod
    def _render_tools_prompt(tools: list[ToolSpec] | None) -> str:
        if not tools:
            return ""
        lines = ["AVAILABLE_TOOLS:"]
        for tool in tools:
            desc = tool.function.description or ""
            lines.append(f"- {tool.function.name}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def _latest_user_content(messages: list[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                return (msg.content or "").strip()
        return ""

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        code_block = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if code_block:
            candidate = code_block.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return None

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return None
        return None

    def _plan_tool_call(self, req: ChatCompletionsRequest) -> ToolCall | None:
        tools = req.tools or []
        if not tools:
            return None
        if req.tool_choice == "none":
            return None

        latest_user = self._latest_user_content(req.messages)
        tool_names = [tool.function.name for tool in tools]
        chosen: str | None = None

        if isinstance(req.tool_choice, ToolChoiceObject):
            if req.tool_choice.function.name not in tool_names:
                raise ValueError(f"Requested tool '{req.tool_choice.function.name}' is not in tools.")
            chosen = req.tool_choice.function.name
        elif req.tool_choice == "required":
            chosen = tool_names[0]
        else:
            lower_user = latest_user.lower()
            for name in tool_names:
                if name.lower() in lower_user:
                    chosen = name
                    break
            if chosen is None:
                return None

        args = self._extract_json_object(latest_user)
        if args is None:
            args = json.dumps({"input": latest_user}, ensure_ascii=False)
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:20]}",
            function=ToolCallFunction(name=chosen, arguments=args),
        )

    @staticmethod
    def _thinking_requested(req: ChatCompletionsRequest) -> bool:
        return bool(req.thinking and req.thinking.enabled)

    @staticmethod
    def _build_thinking(
        req: ChatCompletionsRequest,
        *,
        planned_tool_call: ToolCall | None,
        generated_text: str | None = None,
    ) -> tuple[str | None, int]:
        if not InferenceService._thinking_requested(req):
            return None, 0
        effort = req.reasoning_effort or "medium"
        if planned_tool_call is not None:
            summary = (
                f"Detected tool-use intent and selected function "
                f"'{planned_tool_call.function.name}' with reasoning_effort={effort}."
            )
        else:
            user_input = InferenceService._latest_user_content(req.messages)
            summary = f"Answered directly from model context with reasoning_effort={effort}."
            if user_input:
                snippet = user_input[:80].strip()
                summary = f"{summary} User focus: {snippet}"
            if generated_text:
                summary = f"{summary} Response generated successfully."
        return summary, max(1, len(summary) // 4)

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
