from __future__ import annotations

import json
import re
import uuid

from amlx.schemas import ChatCompletionsRequest, ChatMessage, ToolCall, ToolCallFunction


class InferenceHelpersMixin:
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
        return "\n".join(self._message_to_prompt_line(m) for m in messages)

    @staticmethod
    def _latest_user_content(messages: list[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                return (msg.content or "").strip()
        return ""

    _THINKING_HEADER = re.compile(
        r"^\s*(?:Thinking Process|My Thinking|Chain[- ]of[- ]Thought|Reasoning Process)\s*:\s*\n",
        re.IGNORECASE,
    )

    def _build_prompt(self, req: ChatCompletionsRequest) -> str:
        tokenizer = getattr(self.adapter, "get_tokenizer", lambda _: None)(req.model)
        if tokenizer is None:
            return self._render_prompt(req.messages)
        apply = getattr(tokenizer, "apply_chat_template", None)
        if apply is None:
            return self._render_prompt(req.messages)
        msgs = [{"role": m.role, "content": m.content or ""} for m in req.messages]
        tools_json = None
        if req.tools:
            tools_json = [
                {
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "description": t.function.description or "",
                        "parameters": t.function.parameters or {},
                    },
                }
                for t in req.tools
            ]
        try:
            if tools_json:
                try:
                    return apply(msgs, tools=tools_json, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass
            return apply(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return self._render_prompt(req.messages)

    @staticmethod
    def _parse_tool_call(text: str) -> ToolCall | None:
        # Qwen / hermes format: <tool_call>\n{...}\n</tool_call>
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                name = obj.get("name") or obj.get("function")
                args = obj.get("arguments") or obj.get("parameters") or {}
                if name:
                    return ToolCall(
                        id=f"call_{uuid.uuid4().hex[:20]}",
                        function=ToolCallFunction(
                            name=str(name),
                            arguments=json.dumps(args, ensure_ascii=False),
                        ),
                    )
            except Exception:
                pass
        return None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # XML-style tags (DeepSeek-R1, QwQ standard format)
        text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<think(?:ing)?>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)

        if not InferenceHelpersMixin._THINKING_HEADER.match(text):
            return text.strip()

        original = text

        # Strip "Wait, looking at..." hallucination loops the model gets stuck in
        text = re.sub(
            r"\n+Wait,?\s*(?:looking at|I need to|let me re-read|looking again).*$",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # 1. Explicit "Final Output / Answer / Response:" marker
        m = re.search(r"Final\s+(?:Output|Answer|Response)[^:]*:\s*\*{0,2}\s*(.+)", text, re.IGNORECASE)
        if m:
            answer = m.group(1).strip().split("\n")[0].strip()
            answer = re.sub(r"\s*\(.*\)\s*$", "", answer).strip().rstrip(".")
            if answer:
                return answer

        # 2. Calculation result line: "Calculate: X = Y"
        m = re.search(
            r"(?:Calculate|Compute|Result)[:\s]+[^=]+=\s*(.+?)\.?\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            result = m.group(1).strip()
            if result and len(result) < 120:
                return result

        # 3. Never return empty — fall back to full original text
        return original

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
        if not InferenceHelpersMixin._thinking_requested(req):
            return None, 0
        effort = req.reasoning_effort or "medium"
        if planned_tool_call is not None:
            summary = (
                f"Detected tool-use intent and selected function "
                f"'{planned_tool_call.function.name}' with reasoning_effort={effort}."
            )
        else:
            user_input = InferenceHelpersMixin._latest_user_content(req.messages)
            summary = f"Answered directly from model context with reasoning_effort={effort}."
            if user_input:
                summary = f"{summary} User focus: {user_input[:80].strip()}"
            if generated_text:
                summary = f"{summary} Response generated successfully."
        return summary, max(1, len(summary) // 4)
