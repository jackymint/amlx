from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from amlx.models import ModelManager
from amlx.schemas import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ModelDownloadRequest,
    ModelTrainSaveRequest,
    ModelTrainRequest,
    RuntimePowerRequest,
)
from amlx.service import InferenceService


def create_app(
    service: InferenceService,
    *,
    default_model: str | None = None,
    model_manager: ModelManager | None = None,
) -> FastAPI:
    def resolve_model_ref(model_ref: str) -> str:
        if model_manager is None:
            return model_ref
        try:
            installed = model_manager.installed_models()
        except Exception:
            return model_ref
        for item in installed:
            if item.get("model_id") == model_ref and item.get("path"):
                return str(item["path"])
        return model_ref

    def display_model_ref(model_ref: str) -> str:
        if model_manager is None:
            return model_ref
        try:
            installed = model_manager.installed_models()
        except Exception:
            return model_ref
        for item in installed:
            if item.get("path") == model_ref and item.get("model_id"):
                return str(item["model_id"])
        return model_ref

    def adapter_gpu_limit_state() -> dict[str, object]:
        try:
            return service.adapter.gpu_limit_state()
        except Exception as exc:
            return {"supported": False, "error": str(exc)}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.scheduler.close()

    app = FastAPI(title="amlx", version="0.1.0", lifespan=lifespan)
    ui_dir = files("amlx").joinpath("ui")
    app.mount("/assets", StaticFiles(directory=str(ui_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(str(ui_dir.joinpath("index.html")))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/runtime")
    def runtime() -> dict[str, object]:
        loaded = [display_model_ref(m) for m in service.adapter.loaded_models()]
        return {
            "default_model": default_model,
            "configured_model": default_model,
            "loaded_models": loaded,
            "loaded_default_model": service.adapter.is_model_loaded(default_model) if default_model else False,
            "gpu_limit_percent": service.scheduler.gpu_limit_percent(),
            "gpu_limit_adapter": adapter_gpu_limit_state(),
        }

    @app.get("/v1/runtime/power")
    def runtime_power() -> dict[str, object]:
        return {
            "gpu_limit_percent": service.scheduler.gpu_limit_percent(),
            "gpu_limit_adapter": adapter_gpu_limit_state(),
        }

    @app.post("/v1/runtime/power")
    def runtime_power_update(req: RuntimePowerRequest) -> dict[str, object]:
        current_gpu = service.scheduler.gpu_limit_percent()
        if req.gpu_limit_percent is not None:
            current_gpu = service.scheduler.set_gpu_limit_percent(req.gpu_limit_percent)
            service.adapter.set_gpu_limit_percent(current_gpu)
        return {
            "gpu_limit_percent": current_gpu,
            "gpu_limit_adapter": adapter_gpu_limit_state(),
        }

    @app.get("/v1/models/catalog")
    def models_catalog(page: int = 1, per_page: int = 5) -> dict[str, object]:
        if model_manager is None:
            return {"models": [], "system": {}, "pagination": {"page": 1, "per_page": per_page, "total": 0}}
        models, total = model_manager.catalog(page=page, per_page=per_page)
        return {
            "models": models,
            "system": model_manager.system_profile(),
            "pagination": {"page": page, "per_page": per_page, "total": total},
        }

    @app.get("/v1/models/search")
    def models_search(q: str, page: int = 1, per_page: int = 5) -> dict[str, object]:
        if model_manager is None:
            return {"models": [], "system": {}, "pagination": {"page": 1, "per_page": per_page, "total": 0}}
        models, total = model_manager.search_online(q, page=page, per_page=per_page)
        return {
            "models": models,
            "system": model_manager.system_profile(),
            "pagination": {"page": page, "per_page": per_page, "total": total},
        }

    @app.get("/v1/models/installed")
    def models_installed() -> dict[str, list[dict[str, str]]]:
        if model_manager is None:
            return {"models": []}
        return {"models": model_manager.installed_models()}

    @app.get("/v1/models/downloads")
    def models_downloads() -> dict[str, list[dict[str, str | int | float | None]]]:
        if model_manager is None:
            return {"tasks": []}
        return {"tasks": model_manager.list_tasks()}

    @app.get("/v1/models/downloads/{task_id}")
    def models_download_task(task_id: str) -> dict[str, str | int | float | None]:
        if model_manager is None:
            raise HTTPException(status_code=404, detail="Model manager unavailable")
        task = model_manager.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.post("/v1/models/download")
    def models_download(req: ModelDownloadRequest) -> dict[str, str | int | float | None]:
        if model_manager is None:
            raise HTTPException(status_code=404, detail="Model manager unavailable")
        try:
            return model_manager.enqueue_download(req.model_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/models/preload")
    def models_preload(req: ModelDownloadRequest) -> dict[str, object]:
        try:
            effective_model = resolve_model_ref(req.model_id)
            loaded = service.preload_model(effective_model)
            return {
                "ok": True,
                "loaded": loaded,
                "model_id": req.model_id,
                "effective_model": effective_model,
                "loaded_models": [display_model_ref(m) for m in service.adapter.loaded_models()],
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/models/unload")
    def models_unload(req: ModelDownloadRequest) -> dict[str, object]:
        try:
            effective_model = resolve_model_ref(req.model_id)
            unloaded = service.unload_model(effective_model)
            return {
                "ok": True,
                "unloaded": unloaded,
                "model_id": req.model_id,
                "effective_model": effective_model,
                "loaded_models": [display_model_ref(m) for m in service.adapter.loaded_models()],
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/models/training")
    def models_training() -> dict[str, list[dict[str, object]]]:
        if model_manager is None:
            return {"tasks": []}
        tasks = model_manager.list_finetune_tasks()
        out = []
        for task in tasks:
            row = dict(task)
            row["model_id"] = display_model_ref(str(task.get("model_id", "")))
            out.append(row)
        return {"tasks": out}

    @app.post("/v1/models/train")
    def models_train(req: ModelTrainRequest) -> dict[str, object]:
        if model_manager is None:
            raise HTTPException(status_code=404, detail="Model manager unavailable")
        try:
            effective_model = resolve_model_ref(req.model_id)
            raw = list(req.samples or [])
            if req.dataset_text:
                raw.extend(line.strip() for line in req.dataset_text.splitlines() if line.strip())
            task = model_manager.enqueue_finetune(
                model_id=req.model_id,
                effective_model=effective_model,
                profile=req.profile,
                samples=raw,
                epochs=req.epochs,
                fine_tune_type=req.fine_tune_type,
            )
            return {"ok": True, **task}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/models/train/save")
    def models_train_save(req: ModelTrainSaveRequest) -> dict[str, object]:
        if model_manager is None:
            raise HTTPException(status_code=404, detail="Model manager unavailable")
        try:
            task = model_manager.save_merged_finetune(
                task_id=req.task_id,
                profile=req.profile,
                adapter_path=req.adapter_path,
                effective_model=req.effective_model,
                output_path=req.output_path,
            )
            return {"ok": True, **task}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/models/delete")
    def models_delete(req: ModelDownloadRequest) -> dict[str, object]:
        if model_manager is None:
            raise HTTPException(status_code=404, detail="Model manager unavailable")
        try:
            effective_model = resolve_model_ref(req.model_id)
            unloaded = service.unload_model(effective_model)
            deleted = model_manager.remove_installed_model(req.model_id)
            return {
                "ok": True,
                "deleted": deleted,
                "unloaded": unloaded,
                "model_id": req.model_id,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/cache")
    def cache_clear() -> dict[str, object]:
        service.cache.clear()
        return {"ok": True}

    @app.get("/v1/cache/stats")
    def cache_stats() -> dict[str, int]:
        return {
            "memory_hits": service.cache.stats.memory_hits,
            "disk_hits": service.cache.stats.disk_hits,
            "block_hits": service.cache.stats.block_hits,
            "misses": service.cache.stats.misses,
            "block_writes": service.cache.stats.block_writes,
            "scheduler_enqueued": service.scheduler.stats.enqueued,
            "scheduler_processed": service.scheduler.stats.processed,
            "scheduler_batch_runs": service.scheduler.stats.batch_runs,
            "scheduler_total_batch_items": service.scheduler.stats.total_batch_items,
            "scheduler_throttle_sleep_ms": service.scheduler.stats.throttle_sleep_ms,
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
    def chat_completions(req: ChatCompletionsRequest) -> ChatCompletionsResponse:
        try:
            effective_model = resolve_model_ref(req.model)
            if model_manager is not None:
                adapter_path = model_manager.latest_completed_adapter(
                    model_id=req.model,
                    effective_model=effective_model,
                )
                service.adapter.set_adapter_path(effective_model, adapter_path)
            if effective_model != req.model:
                req = req.model_copy(update={"model": effective_model})
            return service.complete(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
