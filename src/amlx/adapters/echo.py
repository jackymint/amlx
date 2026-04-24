from __future__ import annotations

from threading import Lock

from amlx.adapters.base import GenerationResult, ModelAdapter


class EchoAdapter(ModelAdapter):
    """Safe default adapter for local development before MLX runtime is wired."""

    def __init__(self) -> None:
        self._loaded: set[str] = set()
        self._lock = Lock()

    def loaded_models(self) -> list[str]:
        with self._lock:
            return sorted(self._loaded)

    def preload_model(self, model: str) -> bool:
        with self._lock:
            self._loaded.add(model)
        return True

    def unload_model(self, model: str) -> bool:
        with self._lock:
            if model in self._loaded:
                self._loaded.remove(model)
                return True
        return False

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
