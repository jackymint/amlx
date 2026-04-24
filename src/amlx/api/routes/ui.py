from __future__ import annotations

from importlib.resources import files

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def register_ui_routes(app: FastAPI) -> None:
    ui_dir = files("amlx").joinpath("ui")
    app.mount("/assets", StaticFiles(directory=str(ui_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(str(ui_dir.joinpath("index.html")))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}
