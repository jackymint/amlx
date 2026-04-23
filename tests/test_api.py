from pathlib import Path
from tempfile import mkdtemp
from time import sleep

from fastapi.testclient import TestClient

from amlx.adapters.echo import EchoAdapter
from amlx.api.app import create_app
from amlx.cache.blocks import PagedBlockStore
from amlx.cache.disk import DiskCache
from amlx.cache.memory import LRUCache
from amlx.cache.prefix import PrefixCache
from amlx.models import ModelManager
from amlx.scheduler import BatchScheduler
from amlx.service import InferenceService


def build_client() -> TestClient:
    cache_dir = Path(mkdtemp(prefix="amlx-test-cache-"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = PrefixCache(
        memory_cache=LRUCache(capacity=16),
        disk_cache=DiskCache(db_path=cache_dir / "test_prefix.sqlite3"),
        block_store=PagedBlockStore(
            root_dir=cache_dir / "blocks",
            index_db=cache_dir / "blocks.sqlite3",
            block_chars=256,
        ),
    )
    scheduler = BatchScheduler(adapter=EchoAdapter(), max_batch_size=8, max_wait_ms=20)
    service = InferenceService(adapter=EchoAdapter(), cache=cache, scheduler=scheduler)
    models_dir = Path(mkdtemp(prefix="amlx-test-models-"))

    def fake_downloader(model_id: str, target_dir: Path) -> Path:
        out = models_dir / target_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "weights.safetensors").write_text(f"fake {model_id}", encoding="utf-8")
        return out

    def fake_search(query: str, limit: int) -> list[dict[str, str]]:
        del limit
        return [
            {
                "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                "label": "Qwen2.5 Coder 7B Instruct 4bit",
                "tags": f"online {query}",
                "size": "unknown",
            }
        ]

    manager = ModelManager(models_dir=models_dir, downloader=fake_downloader, search_provider=fake_search)
    return TestClient(create_app(service, model_manager=manager))


def test_health() -> None:
    client = build_client()
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ui_root_and_runtime() -> None:
    client = build_client()
    root = client.get("/")
    assert root.status_code == 200
    assert "amlx Control Room" in root.text

    runtime = client.get("/v1/runtime")
    assert runtime.status_code == 200
    body = runtime.json()
    assert body["default_model"] is None
    assert body["configured_model"] is None
    assert body["engine"] == "echo"
    assert body["loaded_default_model"] is False


def test_chat_completion_and_cache() -> None:
    client = build_client()
    payload = {
        "model": "qwen2.5-coder:7b",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "temperature": 0.2,
    }

    first = client.post("/v1/chat/completions", json=payload)
    assert first.status_code == 200
    data1 = first.json()
    assert data1["choices"][0]["message"]["role"] == "assistant"

    second = client.post("/v1/chat/completions", json=payload)
    assert second.status_code == 200

    stats = client.get("/v1/cache/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["memory_hits"] >= 1
    assert body["scheduler_processed"] >= 1


def test_model_download_flow() -> None:
    client = build_client()
    catalog = client.get("/v1/models/catalog")
    assert catalog.status_code == 200
    model_id = catalog.json()["models"][0]["id"]

    create = client.post("/v1/models/download", json={"model_id": model_id})
    assert create.status_code == 200
    task_id = create.json()["task_id"]

    for _ in range(120):
        status = client.get(f"/v1/models/downloads/{task_id}")
        assert status.status_code == 200
        body = status.json()
        if body["status"] in {"completed", "failed"}:
            break
        sleep(0.05)
    assert body["status"] == "completed"

    installed = client.get("/v1/models/installed")
    assert installed.status_code == 200
    assert any(item["model_id"] == model_id for item in installed.json()["models"])

    unload = client.post("/v1/models/unload", json={"model_id": model_id})
    assert unload.status_code == 200
    assert unload.json()["ok"] is True

    delete = client.post("/v1/models/delete", json={"model_id": model_id})
    assert delete.status_code == 200
    assert delete.json()["ok"] is True
    assert delete.json()["deleted"] is True

    installed_after = client.get("/v1/models/installed")
    assert installed_after.status_code == 200
    assert not any(item["model_id"] == model_id for item in installed_after.json()["models"])


def test_model_catalog_search_and_compatibility() -> None:
    client = build_client()
    res = client.get("/v1/models/search?q=qwen&page=1&per_page=5")
    assert res.status_code == 200
    body = res.json()
    assert "system" in body
    assert "pagination" in body
    assert body["pagination"]["page"] == 1
    assert body["pagination"]["per_page"] == 5
    assert isinstance(body["models"], list)
    assert body["models"]
    assert all("qwen" in item["id"].lower() or "qwen" in item["label"].lower() for item in body["models"])
    first = body["models"][0]
    assert "compatibility" in first
    assert first["compatibility"]["fit"] in {"good", "tight", "no"}
