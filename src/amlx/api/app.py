from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from amlx.api.context import ApiContext
from amlx.api.routes import (
    register_cache_and_chat_routes,
    register_model_ops_routes,
    register_models_catalog_routes,
    register_runtime_routes,
    register_ui_routes,
)
from amlx.models import ModelManager
from amlx.service import InferenceService


def create_app(
    service: InferenceService,
    *,
    default_model: str | None = None,
    model_manager: ModelManager | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.scheduler.close()

    app = FastAPI(title="amlx", version="0.1.0", lifespan=lifespan)
    ctx = ApiContext(
        service=service,
        default_model=default_model,
        model_manager=model_manager,
    )

    register_ui_routes(app)
    register_runtime_routes(app, ctx)
    register_models_catalog_routes(app, ctx)
    register_model_ops_routes(app, ctx)
    register_cache_and_chat_routes(app, ctx)
    return app
