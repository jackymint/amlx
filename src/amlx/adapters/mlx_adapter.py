from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from amlx.adapters.base import GenerationResult, ModelAdapter
from amlx.model_type import is_vlm


@dataclass(slots=True)
class _LoadedModel:
    thread_id: int
    lm: object
    tokenizer: object
    adapter_path: str | None = None


class MLXAdapter(ModelAdapter):
    """Optional runtime adapter.

    This keeps imports local so amlx can run without mlx-lm installed.
    """

    def __init__(self) -> None:
        try:
            from mlx_lm import load  # type: ignore
            import mlx.core as mx  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "mlx-lm is not installed. Install with `pip install 'amlx[mlx]'`."
            ) from exc

        self._load = load
        self._mx = mx
        self._loaded: dict[str, _LoadedModel] = {}
        self._adapter_paths: dict[str, str | None] = {}
        self._gpu_limit_percent = 100
        self._gpu_limit_state: dict[str, object] = {"supported": False}
        self._apply_gpu_limit()

    def loaded_models(self) -> list[str]:
        return sorted(self._loaded.keys())

    def preload_model(self, model: str) -> bool:
        self._ensure_loaded_for_current_thread(model)
        return True

    def unload_model(self, model: str) -> bool:
        if model in self._loaded:
            del self._loaded[model]
            return True
        return False

    def set_adapter_path(self, model: str, adapter_path: str | None) -> bool:
        normalized: str | None = None
        if adapter_path:
            p = Path(adapter_path).expanduser()
            normalized = str(p)
        current = self._adapter_paths.get(model)
        self._adapter_paths[model] = normalized
        changed = current != normalized
        if changed and model in self._loaded:
            del self._loaded[model]
        return changed

    def _ensure_loaded_for_current_thread(self, model: str) -> tuple[object, object]:
        current_tid = threading.get_ident()
        target_adapter = self._adapter_paths.get(model)
        entry = self._loaded.get(model)
        if entry is not None and entry.thread_id == current_tid and entry.adapter_path == target_adapter:
            return entry.lm, entry.tokenizer

        if is_vlm(model):
            raise ValueError(
                f"'{model}' is a vision-language model (VLM) and is not supported. "
                "Use a text-only model instead."
            )

        lm, tokenizer = self._load(model)
        if target_adapter:
            load_adapters = None
            for module_name in ("mlx_lm", "mlx_lm.utils", "mlx_lm.tuner.utils", "mlx_lm.lora"):
                try:
                    module = __import__(module_name, fromlist=["load_adapters"])
                    candidate = getattr(module, "load_adapters", None)
                    if callable(candidate):
                        load_adapters = candidate
                        break
                except Exception:
                    continue
            if load_adapters is None:
                raise RuntimeError(
                    "Failed to import load_adapters for LoRA adapters from mlx_lm."
                )
            load_adapters(lm, target_adapter)
        self._loaded[model] = _LoadedModel(
            thread_id=current_tid,
            lm=lm,
            tokenizer=tokenizer,
            adapter_path=target_adapter,
        )
        return lm, tokenizer

    @staticmethod
    def _clamp_gpu_limit(value: int) -> int:
        return max(20, min(100, int(value)))

    def set_gpu_limit_percent(self, value: int) -> dict[str, object]:
        self._gpu_limit_percent = self._clamp_gpu_limit(value)
        self._apply_gpu_limit()
        return dict(self._gpu_limit_state)

    def gpu_limit_state(self) -> dict[str, object]:
        return dict(self._gpu_limit_state)

    def _apply_gpu_limit(self) -> None:
        state: dict[str, object] = {
            "supported": False,
            "mode": "mlx-memory-cap",
            "requested_percent": self._gpu_limit_percent,
            "applied_percent": self._gpu_limit_percent,
        }
        try:
            mx = self._mx
            info = mx.device_info()
            recommended = int(info.get("max_recommended_working_set_size") or 0)
            memory_size = int(info.get("memory_size") or 0)
            if recommended <= 0:
                state["reason"] = "No recommended working set information"
                self._gpu_limit_state = state
                return

            target = int(recommended * (self._gpu_limit_percent / 100.0))
            min_limit = 512 * 1024 * 1024  # 512MB floor to avoid unusable setting
            target = max(min_limit, target)
            if memory_size > 256 * 1024 * 1024:
                target = min(target, memory_size - (256 * 1024 * 1024))

            mx.set_memory_limit(target)
            mx.set_cache_limit(target)

            wired_target = max(min_limit, int(target * 0.9))
            wired_applied = False
            try:
                mx.set_wired_limit(wired_target)
                wired_applied = True
            except Exception:
                wired_applied = False

            state.update(
                {
                    "supported": True,
                    "recommended_bytes": recommended,
                    "memory_size_bytes": memory_size,
                    "memory_limit_bytes": target,
                    "cache_limit_bytes": target,
                    "wired_limit_bytes": wired_target if wired_applied else None,
                    "wired_limit_applied": wired_applied,
                }
            )
        except Exception as exc:
            state["error"] = str(exc)
        self._gpu_limit_state = state

    def generate(self, *, model: str, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        from mlx_lm import generate as mlx_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        lm, tokenizer = self._ensure_loaded_for_current_thread(model)
        try:
            out = mlx_generate(
                lm,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
            )
        except TypeError:
            out = mlx_generate(
                lm,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        text = out if isinstance(out, str) else str(out)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
