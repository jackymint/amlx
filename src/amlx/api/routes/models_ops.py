from __future__ import annotations

from fastapi import FastAPI, HTTPException

from amlx.api.context import ApiContext
from amlx.schemas import ModelDownloadRequest, ModelQuantizeRequest, ModelTrainRequest, ModelTrainSaveRequest


def register_model_ops_routes(app: FastAPI, ctx: ApiContext) -> None:
    @app.post("/v1/models/download")
    def models_download(req: ModelDownloadRequest) -> dict[str, str | int | float | None]:
        manager = ctx.require_model_manager()
        try:
            return manager.enqueue_download(req.model_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/models/preload")
    def models_preload(req: ModelDownloadRequest) -> dict[str, object]:
        try:
            effective_model = ctx.resolve_model_ref(req.model_id)
            loaded = ctx.service.preload_model(effective_model)
            return {
                "ok": True,
                "loaded": loaded,
                "model_id": req.model_id,
                "effective_model": effective_model,
                "loaded_models": [ctx.display_model_ref(m) for m in ctx.service.adapter.loaded_models()],
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/models/unload")
    def models_unload(req: ModelDownloadRequest) -> dict[str, object]:
        try:
            effective_model = ctx.resolve_model_ref(req.model_id)
            unloaded = ctx.service.unload_model(effective_model)
            return {
                "ok": True,
                "unloaded": unloaded,
                "model_id": req.model_id,
                "effective_model": effective_model,
                "loaded_models": [ctx.display_model_ref(m) for m in ctx.service.adapter.loaded_models()],
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/models/training")
    def models_training() -> dict[str, list[dict[str, object]]]:
        if ctx.model_manager is None:
            return {"tasks": []}
        tasks = ctx.model_manager.list_finetune_tasks()
        out = []
        for task in tasks:
            row = dict(task)
            row["model_id"] = ctx.display_model_ref(str(task.get("model_id", "")))
            out.append(row)
        return {"tasks": out}

    @app.post("/v1/models/train")
    def models_train(req: ModelTrainRequest) -> dict[str, object]:
        manager = ctx.require_model_manager()
        try:
            effective_model = ctx.resolve_model_ref(req.model_id)
            raw = list(req.samples or [])
            if req.dataset_text:
                raw.extend(line.strip() for line in req.dataset_text.splitlines() if line.strip())
            task = manager.enqueue_finetune(
                model_id=req.model_id,
                effective_model=effective_model,
                profile=req.profile,
                samples=raw,
                epochs=req.epochs,
                fine_tune_type=req.fine_tune_type,
                learning_rate=req.learning_rate,
                lora_rank=req.lora_rank,
                lora_layers=req.lora_layers,
                max_seq_length=req.max_seq_length,
                batch_size=req.batch_size,
            )
            return {"ok": True, **task}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/models/train/save")
    def models_train_save(req: ModelTrainSaveRequest) -> dict[str, object]:
        manager = ctx.require_model_manager()
        try:
            task = manager.save_merged_finetune(
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
        manager = ctx.require_model_manager()
        try:
            effective_model = ctx.resolve_model_ref(req.model_id)
            unloaded = ctx.service.unload_model(effective_model)
            deleted = manager.remove_installed_model(req.model_id)
            return {"ok": True, "deleted": deleted, "unloaded": unloaded, "model_id": req.model_id}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/models/quantize")
    def models_quantize_list() -> dict[str, list[dict[str, object]]]:
        if ctx.model_manager is None:
            return {"tasks": []}
        return {"tasks": ctx.model_manager.list_quantize_tasks()}

    @app.post("/v1/models/quantize")
    def models_quantize(req: ModelQuantizeRequest) -> dict[str, object]:
        manager = ctx.require_model_manager()
        try:
            effective_model = ctx.resolve_model_ref(req.model_id)
            task = manager.enqueue_quantize(
                model_id=req.model_id,
                effective_model=effective_model,
                output_path=req.output_path,
                q_bits=req.q_bits,
                q_group_size=req.q_group_size,
            )
            return {"ok": True, **task}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/models/quantize/{task_id}/cancel")
    def models_quantize_cancel(task_id: str) -> dict[str, object]:
        manager = ctx.require_model_manager()
        ok = manager.cancel_quantize(task_id)
        return {"ok": ok, "task_id": task_id}
