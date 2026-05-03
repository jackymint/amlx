from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from amlx.api.context import ApiContext
from amlx.schemas import ChatCompletionsRequest, ChatCompletionsResponse


def register_cache_and_chat_routes(app: FastAPI, ctx: ApiContext) -> None:
    @app.delete("/v1/cache")
    def cache_clear() -> dict[str, object]:
        ctx.service.cache.clear()
        return {"ok": True}

    @app.get("/v1/cache/stats")
    def cache_stats() -> dict[str, int]:
        return {
            "memory_hits": ctx.service.cache.stats.memory_hits,
            "disk_hits": ctx.service.cache.stats.disk_hits,
            "block_hits": ctx.service.cache.stats.block_hits,
            "misses": ctx.service.cache.stats.misses,
            "block_writes": ctx.service.cache.stats.block_writes,
            "scheduler_enqueued": ctx.service.scheduler.stats.enqueued,
            "scheduler_processed": ctx.service.scheduler.stats.processed,
            "scheduler_batch_runs": ctx.service.scheduler.stats.batch_runs,
            "scheduler_total_batch_items": ctx.service.scheduler.stats.total_batch_items,
            "scheduler_throttle_sleep_ms": ctx.service.scheduler.stats.throttle_sleep_ms,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionsRequest):
        try:
            effective_model = ctx.resolve_model_ref(req.model)
            if ctx.model_manager is not None:
                adapter_path = ctx.model_manager.latest_completed_adapter(
                    model_id=req.model,
                    effective_model=effective_model,
                )
                ctx.service.adapter.set_adapter_path(effective_model, adapter_path)
            if effective_model != req.model:
                req = req.model_copy(update={"model": effective_model})
            resp = ctx.service.complete(req)
            if not req.stream:
                return resp
            return StreamingResponse(_to_sse(resp), media_type="text/event-stream")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def _to_sse(resp: ChatCompletionsResponse):
    choice = resp.choices[0]
    content = choice.message.content or ""
    chunk = {
        "id": resp.id,
        "object": "chat.completion.chunk",
        "created": resp.created,
        "model": resp.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": choice.finish_reason,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
