from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    model: str | None = None
    cache_dir: Path = Path.home() / ".amlx" / "cache"
    models_dir: Path = Path.home() / ".amlx" / "models"
    max_memory_cache_items: int = 512
    max_batch_size: int = 8
    batch_wait_ms: int = 20
    block_chars: int = 4096
    log_level: str = "info"

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
