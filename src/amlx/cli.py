from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from amlx import __version__
from amlx.adapters.echo import EchoAdapter
from amlx.adapters.mlx_adapter import MLXAdapter
from amlx.api.app import create_app
from amlx.cache.blocks import PagedBlockStore
from amlx.cache.disk import DiskCache
from amlx.cache.memory import LRUCache
from amlx.cache.prefix import PrefixCache
from amlx.config import ServerConfig
from amlx.models import ModelManager
from amlx.scheduler import BatchScheduler
from amlx.service import InferenceService

app = typer.Typer(
    help="amlx: MacBook-first local inference server",
    no_args_is_help=True,
)


@app.command()
def serve(
    model: str | None = typer.Option(None, help="Optional configured model identifier"),
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    cache_dir: Path = typer.Option(Path.home() / ".amlx" / "cache", help="Cache directory"),
    models_dir: Path = typer.Option(Path.home() / ".amlx" / "models", help="Model storage directory"),
    max_memory_cache_items: int = typer.Option(512, help="In-memory cache capacity"),
    max_batch_size: int = typer.Option(8, help="Maximum scheduler batch size"),
    batch_wait_ms: int = typer.Option(20, help="Scheduler max wait before flush (ms)"),
    block_chars: int = typer.Option(4096, help="Paged block size in UTF-8 characters"),
    engine: str = typer.Option("echo", help="Runtime engine: echo or mlx"),
    gpu_limit_percent: int = typer.Option(100, min=20, max=100, help="Cap average GPU duty cycle to reduce heat"),
    log_level: str = typer.Option("info", help="Uvicorn log level"),
) -> None:
    cfg = ServerConfig(
        host=host,
        port=port,
        model=model,
        cache_dir=cache_dir,
        models_dir=models_dir,
        max_memory_cache_items=max_memory_cache_items,
        max_batch_size=max_batch_size,
        batch_wait_ms=batch_wait_ms,
        block_chars=block_chars,
        log_level=log_level,
    )
    cfg.ensure_dirs()

    if engine == "mlx":
        adapter = MLXAdapter()
    else:
        adapter = EchoAdapter()

    adapter.set_gpu_limit_percent(gpu_limit_percent)

    prefix_cache = PrefixCache(
        memory_cache=LRUCache(capacity=cfg.max_memory_cache_items),
        disk_cache=DiskCache(cfg.cache_dir / "prefix_cache.sqlite3"),
        block_store=PagedBlockStore(
            root_dir=cfg.cache_dir / "blocks",
            index_db=cfg.cache_dir / "blocks.sqlite3",
            block_chars=cfg.block_chars,
        ),
    )
    scheduler = BatchScheduler(
        adapter=adapter,
        max_batch_size=cfg.max_batch_size,
        max_wait_ms=cfg.batch_wait_ms,
        gpu_limit_percent=gpu_limit_percent,
    )
    service = InferenceService(adapter=adapter, cache=prefix_cache, scheduler=scheduler)
    model_manager = ModelManager(models_dir=cfg.models_dir)
    web_app = create_app(
        service,
        default_model=cfg.model,
        engine=engine,
        model_manager=model_manager,
    )

    configured = cfg.model if cfg.model else "(none)"
    typer.echo(f"Starting amlx on http://{cfg.host}:{cfg.port} with configured_model={configured} engine={engine}")
    uvicorn.run(
        web_app,
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
        access_log=False,
    )


@app.command()
def version() -> None:
    typer.echo(f"amlx {__version__}")


if __name__ == "__main__":
    app()
