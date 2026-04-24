from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from amlx.api.context import ApiContext


def register_models_catalog_routes(app: FastAPI, ctx: ApiContext) -> None:
    @app.get("/v1/models/catalog")
    def models_catalog(page: int = 1, per_page: int = 5) -> dict[str, object]:
        if ctx.model_manager is None:
            return {"models": [], "system": {}, "pagination": {"page": 1, "per_page": per_page, "total": 0}}
        models, total = ctx.model_manager.catalog(page=page, per_page=per_page)
        return {
            "models": models,
            "system": ctx.model_manager.system_profile(),
            "pagination": {"page": page, "per_page": per_page, "total": total},
        }

    @app.get("/v1/models/search")
    def models_search(q: str, page: int = 1, per_page: int = 5) -> dict[str, object]:
        if ctx.model_manager is None:
            return {"models": [], "system": {}, "pagination": {"page": 1, "per_page": per_page, "total": 0}}
        models, total = ctx.model_manager.search_online(q, page=page, per_page=per_page)
        return {
            "models": models,
            "system": ctx.model_manager.system_profile(),
            "pagination": {"page": page, "per_page": per_page, "total": total},
        }

    @app.get("/v1/models/installed")
    def models_installed() -> dict[str, list[dict[str, str]]]:
        if ctx.model_manager is None:
            return {"models": []}
        return {"models": ctx.model_manager.installed_models()}

    @app.get("/v1/models/info")
    def models_info(model_id: str) -> dict[str, object]:
        if ctx.model_manager is None:
            return {}
        return ctx.model_manager.model_arch_info(model_id)

    @app.get("/v1/models/downloads")
    def models_downloads() -> dict[str, list[dict[str, str | int | float | None]]]:
        if ctx.model_manager is None:
            return {"tasks": []}
        return {"tasks": ctx.model_manager.list_tasks()}

    @app.get("/v1/models/downloads/{task_id}")
    def models_download_task(task_id: str) -> dict[str, str | int | float | None]:
        manager = ctx.require_model_manager()
        task = manager.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.post("/v1/models/downloads/{task_id}/cancel")
    def models_download_cancel(task_id: str) -> dict[str, object]:
        manager = ctx.require_model_manager()
        ok = manager.cancel_download(task_id)
        return {"ok": ok, "task_id": task_id}

    @app.get("/v1/models/downloads/{task_id}/log")
    async def models_download_log(task_id: str) -> StreamingResponse:
        manager = ctx.require_model_manager()
        log_path = manager._download_log_path(task_id)

        async def stream():
            pos = 0
            while True:
                if log_path.exists():
                    text = log_path.read_text(encoding="utf-8")
                    if len(text) > pos:
                        chunk = text[pos:]
                        pos = len(text)
                        for line in chunk.splitlines():
                            if line:
                                yield f"data: {line}\n\n"
                task = manager.get_task(task_id)
                if task and str(task.get("status")) in {"completed", "failed"}:
                    if log_path.exists() and len(log_path.read_text(encoding="utf-8")) <= pos:
                        yield "data: [end]\n\n"
                        break
                await asyncio.sleep(0.3)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
