from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True)
class TrainingResult:
    trained_samples: int
    epochs: int
    message: str


class ModelAdapter:
    def generate(self, *, model: str, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        raise NotImplementedError

    def loaded_models(self) -> list[str]:
        return []

    def is_model_loaded(self, model: str) -> bool:
        return model in self.loaded_models()

    def preload_model(self, model: str) -> bool:
        del model
        return False

    def unload_model(self, model: str) -> bool:
        del model
        return False

    def set_adapter_path(self, model: str, adapter_path: str | None) -> bool:
        del model, adapter_path
        return False

    def train(self, *, model: str, samples: list[str], epochs: int) -> TrainingResult:
        del model, samples, epochs
        raise NotImplementedError("Training is not supported by this adapter.")

    def set_gpu_limit_percent(self, value: int) -> dict[str, object]:
        del value
        return {"supported": False}

    def gpu_limit_state(self) -> dict[str, object]:
        return {"supported": False}

    def generate_batch(
        self,
        *,
        model: str,
        prompts: list[str],
        max_tokens: int,
        temperature: float,
    ) -> list[GenerationResult]:
        return [
            self.generate(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for prompt in prompts
        ]
