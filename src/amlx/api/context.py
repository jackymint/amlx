from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from amlx.models import ModelManager
from amlx.service import InferenceService


@dataclass(slots=True)
class ApiContext:
    service: InferenceService
    default_model: str | None
    model_manager: ModelManager | None

    def require_model_manager(self) -> ModelManager:
        if self.model_manager is None:
            raise HTTPException(status_code=404, detail="Model manager unavailable")
        return self.model_manager

    def resolve_model_ref(self, model_ref: str) -> str:
        if self.model_manager is None:
            return model_ref
        try:
            installed = self.model_manager.installed_models()
        except Exception:
            return model_ref
        for item in installed:
            if item.get("model_id") == model_ref and item.get("path"):
                return str(item["path"])
        return model_ref

    def display_model_ref(self, model_ref: str) -> str:
        if self.model_manager is None:
            return model_ref
        try:
            installed = self.model_manager.installed_models()
        except Exception:
            return model_ref
        for item in installed:
            if item.get("path") == model_ref and item.get("model_id"):
                return str(item["model_id"])
        return model_ref

    def adapter_gpu_limit_state(self) -> dict[str, object]:
        try:
            return self.service.adapter.gpu_limit_state()
        except Exception as exc:
            return {"supported": False, "error": str(exc)}
