from pathlib import Path
from tempfile import mkdtemp
from time import sleep

from fastapi.testclient import TestClient
from mlx_lm.tuner.callbacks import TrainingCallback

from amlx.adapters.echo import EchoAdapter
from amlx.api.app import create_app
from amlx.cache.blocks import PagedBlockStore
from amlx.cache.disk import DiskCache
from amlx.cache.memory import LRUCache
from amlx.cache.prefix import PrefixCache
from amlx.models import ModelManager
from amlx.scheduler import BatchScheduler
from amlx.service import InferenceService


class TrackingEchoAdapter(EchoAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.last_adapter: dict[str, str | None] = {}

    def set_adapter_path(self, model: str, adapter_path: str | None) -> bool:
        prev = self.last_adapter.get(model)
        self.last_adapter[model] = adapter_path
        return prev != adapter_path


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
    adapter = TrackingEchoAdapter()
    scheduler = BatchScheduler(adapter=adapter, max_batch_size=8, max_wait_ms=20)
    service = InferenceService(adapter=adapter, cache=cache, scheduler=scheduler)
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

    def fake_fine_tuner(args, callback: TrainingCallback | None) -> None:
        total = max(1, int(getattr(args, "iters", 20)))
        if callback is not None:
            callback.on_train_loss_report({"iteration": max(1, total // 2), "train_loss": 1.23})
            callback.on_train_loss_report({"iteration": total, "train_loss": 0.42})

    manager = ModelManager(
        models_dir=models_dir,
        downloader=fake_downloader,
        search_provider=fake_search,
        fine_tuner=fake_fine_tuner,
    )
    app = create_app(service, model_manager=manager)
    app.state.test_adapter = adapter
    return TestClient(app)


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
    assert "capabilities" in first
    assert set(first["capabilities"].keys()) == {
        "tool",
        "vision",
        "thinking",
        "coding",
        "embedding",
        "rerank",
        "audio",
        "speech",
    }
    assert isinstance(first["capabilities"]["tool"], bool)


def test_train_requires_loaded_model() -> None:
    client = build_client()
    model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    res = client.post("/v1/models/train", json={"model_id": model_id, "dataset_text": "Use concise style."})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["status"] in {"queued", "running", "completed"}
    assert body["fine_tune_type"] == "qlora"


def test_train_loaded_model_flow() -> None:
    client = build_client()
    model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    train = client.post(
        "/v1/models/train",
        json={
            "model_id": model_id,
            "dataset_text": "Respond in Thai.\nMention cache behavior when relevant.",
            "epochs": 2,
            "fine_tune_type": "qlora",
        },
    )
    assert train.status_code == 200
    body = train.json()
    assert body["ok"] is True
    task_id = body["task_id"]

    final = None
    for _ in range(40):
        trained = client.get("/v1/models/training")
        assert trained.status_code == 200
        entries = trained.json()["tasks"]
        match = [item for item in entries if item["task_id"] == task_id]
        if match:
            final = match[0]
            if final["status"] in {"completed", "failed"}:
                break
        sleep(0.05)
    assert final is not None
    assert final["status"] == "completed"
    assert final["fine_tune_type"] == "qlora"
    assert final["train_samples"] == 2
    assert final["adapter_path"]


def test_chat_auto_applies_latest_completed_adapter() -> None:
    client = build_client()
    model_id = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    train = client.post(
        "/v1/models/train",
        json={
            "model_id": model_id,
            "samples": ["When user asks weather, call get_weather with {\"city\":\"Bangkok\"}"],
            "epochs": 1,
            "fine_tune_type": "qlora",
        },
    )
    assert train.status_code == 200
    task_id = train.json()["task_id"]

    final = None
    for _ in range(40):
        trained = client.get("/v1/models/training")
        assert trained.status_code == 200
        entries = trained.json()["tasks"]
        match = [item for item in entries if item["task_id"] == task_id]
        if match:
            final = match[0]
            if final["status"] == "completed":
                break
        sleep(0.05)
    assert final is not None
    assert final["status"] == "completed"
    adapter_path = final["adapter_path"]
    assert adapter_path

    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "temperature": 0.2,
        },
    )
    assert chat.status_code == 200
    assert client.app.state.test_adapter.last_adapter
    assert client.app.state.test_adapter.last_adapter.get(model_id) == adapter_path


def test_chat_completions_tool_call_required() -> None:
    client = build_client()
    payload = {
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "messages": [
            {"role": "user", "content": 'Please call get_weather for Bangkok {"city":"Bangkok"}'}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city name",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "required",
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    body = res.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert '"city": "Bangkok"' in choice["message"]["tool_calls"][0]["function"]["arguments"]


def test_chat_completions_tool_choice_none_returns_text() -> None:
    client = build_client()
    payload = {
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "messages": [{"role": "user", "content": "hello tool world"}],
        "tools": [{"type": "function", "function": {"name": "search_docs"}}],
        "tool_choice": "none",
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    body = res.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["tool_calls"] is None
    assert choice["message"]["content"]


def test_chat_completions_thinking_text_response() -> None:
    client = build_client()
    payload = {
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "messages": [{"role": "user", "content": "Summarize this in one line"}],
        "thinking": {"enabled": True, "summary": "auto"},
        "reasoning_effort": "medium",
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    body = res.json()
    msg = body["choices"][0]["message"]
    assert msg["thinking"]
    assert "reasoning_effort=medium" in msg["thinking"]
    assert body["usage"]["reasoning_tokens"] >= 1


def test_chat_completions_thinking_tool_call_response() -> None:
    client = build_client()
    payload = {
        "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "messages": [{"role": "user", "content": "run search_docs {\"query\":\"mlx\"}"}],
        "tools": [{"type": "function", "function": {"name": "search_docs"}}],
        "tool_choice": "required",
        "thinking": {"enabled": True, "summary": "detailed"},
        "reasoning_effort": "high",
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    body = res.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["thinking"]
    assert "selected function 'search_docs'" in choice["message"]["thinking"]
    assert body["usage"]["reasoning_tokens"] >= 1
