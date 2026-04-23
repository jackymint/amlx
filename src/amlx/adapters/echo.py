from __future__ import annotations

from amlx.adapters.base import GenerationResult, ModelAdapter


class EchoAdapter(ModelAdapter):
    """Safe default adapter for local development before MLX runtime is wired."""

    def generate(self, *, model: str, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        del temperature
        truncated = prompt[-min(max_tokens, 400):]
        text = f"[amlx:{model}] {truncated}"
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
