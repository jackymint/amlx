from __future__ import annotations

from fastapi import FastAPI

from amlx.api.context import ApiContext
from amlx.schemas import RuntimePowerRequest


def register_runtime_routes(app: FastAPI, ctx: ApiContext) -> None:
    @app.get("/v1/models")
    def list_models() -> dict[str, object]:
        loaded = [ctx.display_model_ref(m) for m in ctx.service.adapter.loaded_models()]
        installed = ctx.model_manager.installed_models() if ctx.model_manager else []
        all_ids = list({*loaded, *[m.get("model_id", "") for m in installed if m.get("model_id")]})
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "amlx",
                    "created": 0,
                }
                for model_id in all_ids
                if model_id
            ],
        }

    @app.get("/v1/runtime")
    def runtime() -> dict[str, object]:
        loaded = [ctx.display_model_ref(m) for m in ctx.service.adapter.loaded_models()]
        return {
            "default_model": ctx.default_model,
            "configured_model": ctx.default_model,
            "loaded_models": loaded,
            "loaded_default_model": ctx.service.adapter.is_model_loaded(ctx.default_model) if ctx.default_model else False,
            "gpu_limit_percent": ctx.service.scheduler.gpu_limit_percent(),
            "gpu_limit_adapter": ctx.adapter_gpu_limit_state(),
        }

    @app.get("/v1/runtime/power")
    def runtime_power() -> dict[str, object]:
        return {
            "gpu_limit_percent": ctx.service.scheduler.gpu_limit_percent(),
            "gpu_limit_adapter": ctx.adapter_gpu_limit_state(),
        }

    @app.post("/v1/runtime/power")
    def runtime_power_update(req: RuntimePowerRequest) -> dict[str, object]:
        current_gpu = ctx.service.scheduler.gpu_limit_percent()
        if req.gpu_limit_percent is not None:
            current_gpu = ctx.service.scheduler.set_gpu_limit_percent(req.gpu_limit_percent)
            ctx.service.adapter.set_gpu_limit_percent(current_gpu)
        return {
            "gpu_limit_percent": current_gpu,
            "gpu_limit_adapter": ctx.adapter_gpu_limit_state(),
        }
