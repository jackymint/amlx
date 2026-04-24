from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep, time
from typing import Any, Callable

from amlx.model_type import is_vlm

CatalogItem = dict[str, Any]
Downloader = Callable[[str, Path], Path]
SearchProvider = Callable[[str, int], list[dict[str, Any]]]
FineTuner = Callable[[Any, Any], None]


CATALOG: list[CatalogItem] = [
    {
        "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "label": "Qwen2.5 Coder 7B 4bit",
        "size": "~4.7 GB",
        "tags": "coding, balanced",
        "disk_gb": 8.0,
        "min_ram_gb": 8.0,
        "rec_ram_gb": 16.0,
        "requires_apple_silicon": True,
    },
    {
        "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "label": "Llama 3.2 3B 4bit",
        "size": "~2.1 GB",
        "tags": "fast, lightweight",
        "disk_gb": 4.0,
        "min_ram_gb": 6.0,
        "rec_ram_gb": 12.0,
        "requires_apple_silicon": True,
    },
    {
        "id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "label": "Mistral 7B Instruct 4bit",
        "size": "~4.4 GB",
        "tags": "general, stable",
        "disk_gb": 8.0,
        "min_ram_gb": 8.0,
        "rec_ram_gb": 16.0,
        "requires_apple_silicon": True,
    },
    {
        "id": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        "label": "DeepSeek R1 Distill Qwen 7B",
        "size": "~4.9 GB",
        "tags": "reasoning, coding",
        "disk_gb": 9.0,
        "min_ram_gb": 12.0,
        "rec_ram_gb": 24.0,
        "requires_apple_silicon": True,
    },
]


@dataclass(slots=True)
class DownloadTask:
    task_id: str
    model_id: str
    status: str
    progress: int
    message: str
    started_at: float
    updated_at: float
    finished_at: float | None = None
    local_path: str | None = None
    error: str | None = None


@dataclass(slots=True)
class FineTuneTask:
    task_id: str
    model_id: str
    effective_model: str
    profile: str
    profile_slug: str
    round: int
    status: str
    progress: int
    message: str
    fine_tune_type: str
    epochs: int
    train_samples: int
    started_at: float
    updated_at: float
    finished_at: float | None = None
    adapter_path: str | None = None
    resume_adapter_file: str | None = None
    run_root: str | None = None
    merged_path: str | None = None
    error: str | None = None


class ModelManager:
    def __init__(
        self,
        *,
        models_dir: Path,
        downloader: Downloader | None = None,
        search_provider: SearchProvider | None = None,
        fine_tuner: FineTuner | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, DownloadTask] = {}
        self._finetune_tasks: dict[str, FineTuneTask] = {}
        self._lock = threading.Lock()
        self._downloader = downloader or self._default_downloader
        self._search_provider = search_provider or self._default_search_provider
        self._fine_tuner = fine_tuner or self._default_fine_tuner

    def system_profile(self) -> dict[str, Any]:
        machine = platform.machine().lower()
        system = platform.system().lower()
        ram_gb = round(self._total_ram_gb(), 1)
        free_disk_gb = round(shutil.disk_usage(self.models_dir).free / (1024**3), 1)
        apple_silicon = system == "darwin" and machine in {"arm64", "aarch64"}
        return {
            "system": system,
            "machine": machine,
            "apple_silicon": apple_silicon,
            "ram_gb": ram_gb,
            "free_disk_gb": free_disk_gb,
            "models_dir": str(self.models_dir),
        }

    def catalog(self, page: int = 1, per_page: int = 12) -> tuple[list[CatalogItem], int]:
        page = max(1, page)
        per_page = max(1, min(per_page, 100))
        profile = self.system_profile()
        enriched_all: list[CatalogItem] = []
        for item in CATALOG:
            enriched = dict(item)
            enriched["compatibility"] = self._compatibility(item, profile)
            enriched["capabilities"] = self._capabilities(enriched)
            enriched_all.append(enriched)
        total = len(enriched_all)
        start = (page - 1) * per_page
        end = start + per_page
        return enriched_all[start:end], total

    def search_online(self, query: str, page: int = 1, per_page: int = 12) -> tuple[list[CatalogItem], int]:
        page = max(1, page)
        per_page = max(1, min(per_page, 100))
        profile = self.system_profile()
        # Pull a broad online result set first, then paginate in UI/API.
        # This avoids the "only 5 found" issue when per_page is set to 5.
        online_limit = max(200, min(1000, per_page * 200))
        online = self._search_provider(query, online_limit)
        total = len(online)
        start = (page - 1) * per_page
        end = start + per_page
        online_page = online[start:end]
        items: list[CatalogItem] = []
        for item in online_page:
            disk_gb, min_ram_gb, rec_ram_gb = self._estimate_requirements(item.get("id", ""))
            enriched: CatalogItem = {
                "id": item.get("id", "unknown"),
                "label": item.get("label") or item.get("id", "unknown"),
                "size": item.get("size", "unknown"),
                "tags": item.get("tags", "online"),
                "disk_gb": disk_gb,
                "min_ram_gb": min_ram_gb,
                "rec_ram_gb": rec_ram_gb,
                "requires_apple_silicon": True,
                "source": "online",
            }
            enriched["compatibility"] = self._compatibility(enriched, profile)
            enriched["capabilities"] = self._capabilities(enriched)
            items.append(enriched)
        return items, total

    @staticmethod
    def _capabilities(item: CatalogItem) -> dict[str, bool]:
        ident = str(item.get("id", "")).lower()
        tags = str(item.get("tags", "")).lower()
        text = f"{ident} {tags}"

        vision_tokens = ("vlm", "vision", "multimodal", "llava", "qwen-vl", "gemma-vision")
        thinking_tokens = ("reasoning", "r1", "think", "thinking", "o1", "o3", "deepseek-r1")
        tool_tokens = ("instruct", "chat", "assistant", "function", "tool", "agent", "coder")
        coding_tokens = ("coder", "coding", "code", "dev", "programming")
        embedding_tokens = ("embedding", "embed")
        rerank_tokens = ("rerank", "reranker")
        audio_tokens = ("audio", "asr", "whisper", "speech", "tts", "stt", "voice")
        speech_tokens = ("speech", "tts", "stt", "voice", "asr", "whisper")

        vision = any(token in text for token in vision_tokens) or is_vlm(ident)
        thinking = any(token in text for token in thinking_tokens)
        tool = any(token in text for token in tool_tokens) and not vision
        coding = any(token in text for token in coding_tokens) and not vision
        embedding = any(token in text for token in embedding_tokens)
        rerank = any(token in text for token in rerank_tokens)
        audio = any(token in text for token in audio_tokens)
        speech = any(token in text for token in speech_tokens)

        return {
            "tool": bool(tool),
            "vision": bool(vision),
            "thinking": bool(thinking),
            "coding": bool(coding),
            "embedding": bool(embedding),
            "rerank": bool(rerank),
            "audio": bool(audio),
            "speech": bool(speech),
        }

    def installed_models(self) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for path in sorted(self.models_dir.glob("*")):
            if not path.is_dir():
                continue
            marker = path / "amlx_model.json"
            if not marker.exists():
                continue
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
                items.append(
                    {
                        "model_id": str(data.get("model_id", "unknown")),
                        "path": str(path),
                    }
                )
            except Exception:
                continue
        return items

    def remove_installed_model(self, model_id: str) -> bool:
        target_path: Path | None = None
        for item in self.installed_models():
            if item.get("model_id") == model_id and item.get("path"):
                target_path = Path(item["path"])
                break

        if target_path is None:
            return False

        root = self.models_dir.resolve()
        try:
            resolved = target_path.resolve()
        except Exception:
            return False

        if not resolved.is_relative_to(root):
            return False

        if not resolved.exists() or not resolved.is_dir():
            return False

        shutil.rmtree(resolved)
        self._remove_finetunes_for_model(model_id)
        return True

    def _remove_finetunes_for_model(self, model_id: str) -> None:
        tasks = self.list_finetune_tasks()
        for task in tasks:
            if str(task.get("model_id") or "") != model_id and str(task.get("effective_model") or "") != model_id:
                continue
            run_root_raw = str(task.get("run_root") or "").strip()
            task_id = str(task.get("task_id") or "").strip()
            run_root = Path(run_root_raw) if run_root_raw else None
            if run_root is None and task_id:
                run_root = self._finetune_root() / task_id
            if run_root and run_root.exists() and run_root.is_dir():
                try:
                    shutil.rmtree(run_root)
                except Exception:
                    pass
            if task_id:
                with self._lock:
                    self._finetune_tasks.pop(task_id, None)

    def list_tasks(self) -> list[dict[str, str | int | float | None]]:
        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda x: x.started_at, reverse=True)
            return [asdict(t) for t in tasks]

    def list_finetune_tasks(self) -> list[dict[str, str | int | float | None]]:
        with self._lock:
            in_memory = {task.task_id: asdict(task) for task in self._finetune_tasks.values()}
        recovered = self._scan_finetune_runs()
        for row in recovered:
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            in_memory.setdefault(task_id, row)
        tasks = list(in_memory.values())
        tasks.sort(
            key=lambda x: float(x.get("started_at") or x.get("updated_at") or 0),
            reverse=True,
        )
        return tasks

    def get_finetune_task(self, task_id: str) -> dict[str, str | int | float | None] | None:
        with self._lock:
            task = self._finetune_tasks.get(task_id)
            if task is None:
                return None
            return asdict(task)

    def latest_completed_adapter(
        self,
        *,
        model_id: str | None = None,
        effective_model: str | None = None,
    ) -> str | None:
        tasks = self.list_finetune_tasks()
        best: dict[str, str | int | float | None] | None = None
        for task in tasks:
            if task.get("status") != "completed":
                continue
            adapter_path = str(task.get("adapter_path") or "").strip()
            if not adapter_path:
                continue
            if model_id and str(task.get("model_id")) == model_id:
                pass
            elif effective_model and str(task.get("effective_model")) == effective_model:
                pass
            else:
                continue
            if best is None:
                best = task
                continue
            best_ts = float(best.get("finished_at") or best.get("updated_at") or 0)
            task_ts = float(task.get("finished_at") or task.get("updated_at") or 0)
            if task_ts >= best_ts:
                best = task
        if best is None:
            return None
        return str(best.get("adapter_path") or "")

    def latest_completed_profile_task(self, *, profile: str) -> dict[str, str | int | float | None] | None:
        profile_name = self._normalize_profile_name(profile, model_id="")
        profile_slug = self._profile_slug(profile_name)
        tasks = self.list_finetune_tasks()
        best: dict[str, str | int | float | None] | None = None
        for task in tasks:
            if task.get("status") != "completed":
                continue
            adapter_path = str(task.get("adapter_path") or "").strip()
            if not adapter_path:
                continue
            task_slug = str(task.get("profile_slug") or "").strip()
            task_profile = str(task.get("profile") or "").strip()
            if task_slug and task_slug == profile_slug:
                pass
            elif task_profile and self._profile_slug(task_profile) == profile_slug:
                pass
            else:
                continue
            if best is None:
                best = task
                continue
            best_ts = float(best.get("finished_at") or best.get("updated_at") or 0)
            task_ts = float(task.get("finished_at") or task.get("updated_at") or 0)
            if task_ts >= best_ts:
                best = task
        return best

    def save_merged_finetune(
        self,
        *,
        task_id: str | None,
        profile: str | None,
        adapter_path: str | None,
        effective_model: str | None,
        output_path: str,
    ) -> dict[str, str | int | float | None]:
        chosen: dict[str, str | int | float | None] | None = None
        resolved_task_id = str(task_id or "").strip()
        if resolved_task_id:
            for row in self.list_finetune_tasks():
                if str(row.get("task_id") or "") != resolved_task_id:
                    continue
                chosen = row
                break
            if chosen is None:
                raise ValueError("Training task not found")
            if str(chosen.get("status") or "") != "completed":
                raise ValueError("Training task is not completed yet")
            if not effective_model:
                effective_model = str(chosen.get("effective_model") or "")
            if not adapter_path:
                adapter_path = str(chosen.get("adapter_path") or "")

        profile_name = str(profile or "").strip()
        if chosen is None and profile_name:
            chosen = self.latest_completed_profile_task(profile=profile_name)
            if chosen is None:
                raise ValueError(f"No completed training found for profile '{profile_name}'")
            if not effective_model:
                effective_model = str(chosen.get("effective_model") or "")
            if not adapter_path:
                adapter_path = str(chosen.get("adapter_path") or "")

        effective_model = str(effective_model or "").strip()
        adapter_path = str(adapter_path or "").strip()
        if not effective_model:
            raise ValueError("Effective model is required")
        if not adapter_path:
            raise ValueError("Adapter path is required")

        target = Path(output_path).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        target = target.resolve()
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Output path is not a directory: {target}")
            if any(target.iterdir()):
                raise ValueError(f"Output path is not empty: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

        self._fuse_adapter_into_model(
            effective_model=effective_model,
            adapter_path=Path(adapter_path),
            target_path=target,
        )

        merged_path = str(target)
        if chosen is not None:
            chosen = dict(chosen)
            chosen["merged_path"] = merged_path
            chosen["updated_at"] = time()
        if resolved_task_id:
            with self._lock:
                current = self._finetune_tasks.get(resolved_task_id)
                if current is not None:
                    current.merged_path = merged_path
                    current.updated_at = time()
                    payload = asdict(current)
                    self._persist_finetune_task(payload)
                    return payload
        if chosen is None:
            chosen = {
                "task_id": resolved_task_id or f"saved_{uuid.uuid4().hex[:12]}",
                "model_id": effective_model,
                "effective_model": effective_model,
                "status": "completed",
                "progress": 100,
                "message": "Merged and saved",
                "fine_tune_type": "qlora",
                "epochs": 0,
                "train_samples": 0,
                "started_at": time(),
                "updated_at": time(),
                "finished_at": time(),
                "adapter_path": adapter_path,
                "merged_path": merged_path,
                "error": None,
            }
        if resolved_task_id:
            run_root_raw = str((chosen or {}).get("run_root") or "").strip()
            run_root = Path(run_root_raw) if run_root_raw else (self._finetune_root() / resolved_task_id)
            run_root.mkdir(parents=True, exist_ok=True)
            self._write_finetune_meta(run_root, chosen)
        return chosen

    def get_task(self, task_id: str) -> dict[str, str | int | float | None] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return asdict(task)

    def enqueue_download(self, model_id: str) -> dict[str, str | int | float | None]:
        if is_vlm(model_id):
            raise ValueError(
                f"'{model_id}' is a vision-language model (VLM) and is not supported."
            )

        existing = self._active_task_for_model(model_id)
        if existing is not None:
            return asdict(existing)

        now = time()
        task = DownloadTask(
            task_id=f"dl_{uuid.uuid4().hex[:18]}",
            model_id=model_id,
            status="queued",
            progress=0,
            message="Queued",
            started_at=now,
            updated_at=now,
        )
        with self._lock:
            self._tasks[task.task_id] = task

        thread = threading.Thread(target=self._run_download, args=(task.task_id,), daemon=True)
        thread.start()
        return asdict(task)

    def enqueue_finetune(
        self,
        *,
        model_id: str,
        effective_model: str,
        profile: str | None,
        samples: list[str],
        epochs: int,
        fine_tune_type: str = "qlora",
    ) -> dict[str, str | int | float | None]:
        clean = [s.strip() for s in samples if s and s.strip()]
        if not clean:
            raise ValueError("Training data is empty.")

        profile_name = self._normalize_profile_name(profile, model_id=model_id)
        profile_slug = self._profile_slug(profile_name)

        existing = self._active_finetune_task_for_profile(profile_slug)
        if existing is not None:
            return asdict(existing)

        previous = self.latest_completed_profile_task(profile=profile_name)
        previous_round = int(previous.get("round") or 0) if previous else 0
        round_no = previous_round + 1
        previous_adapter = str(previous.get("adapter_path") or "").strip() if previous else ""
        resume_adapter_file = self._adapter_weights_file(Path(previous_adapter)) if previous_adapter else None
        task_id = f"ft_{profile_slug}_{uuid.uuid4().hex[:12]}"
        run_root = self._profile_runs_root(profile_slug) / task_id

        now = time()
        task = FineTuneTask(
            task_id=task_id,
            model_id=model_id,
            effective_model=effective_model,
            profile=profile_name,
            profile_slug=profile_slug,
            round=round_no,
            status="queued",
            progress=0,
            message="Queued",
            fine_tune_type=fine_tune_type,
            epochs=max(1, int(epochs)),
            train_samples=len(clean),
            started_at=now,
            updated_at=now,
            resume_adapter_file=resume_adapter_file,
            run_root=str(run_root),
        )
        with self._lock:
            self._finetune_tasks[task.task_id] = task

        worker = threading.Thread(
            target=self._run_finetune,
            args=(task.task_id, clean),
            daemon=True,
        )
        worker.start()
        payload = asdict(task)
        self._persist_finetune_task(payload)
        return payload

    def _active_task_for_model(self, model_id: str) -> DownloadTask | None:
        with self._lock:
            for task in self._tasks.values():
                if task.model_id == model_id and task.status in {"queued", "downloading"}:
                    return task
        return None

    def _active_finetune_task_for_profile(self, profile_slug: str) -> FineTuneTask | None:
        with self._lock:
            for task in self._finetune_tasks.values():
                if task.profile_slug == profile_slug and task.status in {"queued", "running"}:
                    return task
        return None

    def _run_download(self, task_id: str) -> None:
        self._update(task_id, status="downloading", progress=0, message="Preparing download")

        task = self.get_task(task_id)
        if task is None:
            return

        model_id = str(task["model_id"])
        target_rel = self._model_path(model_id)
        target_dir = self.models_dir / target_rel

        try:
            marker = target_dir / "amlx_model.json"
            if marker.exists():
                self._update(
                    task_id,
                    status="completed",
                    progress=100,
                    message="Already installed",
                    local_path=str(target_dir),
                    finished_at=time(),
                )
                return

            self._update(task_id, progress=1, message="Fetching model artifacts")
            expected_bytes = self._estimate_repo_size_bytes(model_id)
            start_bytes = self._dir_size_bytes(target_dir)

            result: dict[str, Path] = {}
            error_holder: list[Exception] = []
            done = threading.Event()

            def run_download() -> None:
                try:
                    result["path"] = self._downloader(model_id, target_rel)
                except Exception as exc:
                    error_holder.append(exc)
                finally:
                    done.set()

            worker = threading.Thread(target=run_download, daemon=True)
            worker.start()

            rolling = 1
            while not done.wait(timeout=1.0):
                if expected_bytes and expected_bytes > 0:
                    current_bytes = max(0, self._dir_size_bytes(target_dir) - start_bytes)
                    ratio = min(1.0, current_bytes / expected_bytes)
                    rolling = max(rolling, min(95, 1 + int(ratio * 94)))
                else:
                    rolling = min(90, rolling + 1)
                self._update(task_id, progress=rolling, message="Fetching model artifacts")

            worker.join(timeout=0.1)
            if error_holder:
                raise error_holder[0]
            path = result["path"]
            self._update(task_id, progress=98, message="Finalizing")

            marker.write_text(
                json.dumps({"model_id": model_id, "downloaded_at": int(time())}),
                encoding="utf-8",
            )

            self._update(
                task_id,
                status="completed",
                progress=100,
                message="Downloaded",
                local_path=str(path),
                finished_at=time(),
            )
        except Exception as exc:
            self._update(
                task_id,
                status="failed",
                progress=100,
                message="Failed",
                error=str(exc),
                finished_at=time(),
            )

    def _run_finetune(self, task_id: str, samples: list[str]) -> None:
        self._update_finetune(task_id, status="running", progress=1, message="Preparing dataset")
        task = self.get_finetune_task(task_id)
        if task is None:
            return

        run_root_raw = str(task.get("run_root") or "").strip()
        run_root = Path(run_root_raw) if run_root_raw else (self._finetune_root() / str(task_id))
        data_dir = run_root / "data"
        adapter_dir = run_root / "adapters"
        data_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_rows = [{"text": row} for row in samples]
            with (data_dir / "train.jsonl").open("w", encoding="utf-8") as f:
                for row in train_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            iters = max(20, int(task["epochs"]) * max(10, len(samples)))
            steps_per_report = max(1, min(20, iters // 20))
            save_every = max(20, iters // 5)
            fine_type = str(task["fine_tune_type"]).lower()
            mlx_fine_type = "lora" if fine_type in {"lora", "qlora"} else "dora"

            import argparse
            from mlx_lm import lora as mlx_lora  # type: ignore
            from mlx_lm.tuner.callbacks import TrainingCallback  # type: ignore

            class _ProgressCallback(TrainingCallback):
                def __init__(self, update_fn: Callable[[int, str], None], total_iters: int):
                    self._update_fn = update_fn
                    self._total_iters = max(1, total_iters)

                def on_train_loss_report(self, train_info: dict):
                    it = int(train_info.get("iteration", 0))
                    pct = max(1, min(99, int((it / self._total_iters) * 100)))
                    loss = train_info.get("train_loss")
                    msg = f"Training (iter {it}/{self._total_iters})"
                    if isinstance(loss, (int, float)):
                        msg = f"{msg} • loss {loss:.4f}"
                    self._update_fn(pct, msg)

                def on_val_loss_report(self, val_info: dict):
                    loss = val_info.get("val_loss")
                    if isinstance(loss, (int, float)):
                        self._update_fn(99, f"Validation • loss {loss:.4f}")

            config = dict(mlx_lora.CONFIG_DEFAULTS)
            config.update(
                {
                    "model": str(task["effective_model"]),
                    "train": True,
                    "test": False,
                    "data": str(data_dir),
                    "fine_tune_type": mlx_fine_type,
                    "optimizer": "adamw",
                    "batch_size": 1,
                    "iters": iters,
                    "val_batches": 1,
                    "learning_rate": 1e-5,
                    "steps_per_report": steps_per_report,
                    "steps_per_eval": max(iters + 1, 5000),
                    "grad_accumulation_steps": 1,
                    "resume_adapter_file": task.get("resume_adapter_file"),
                    "adapter_path": str(adapter_dir),
                    "save_every": save_every,
                    "max_seq_length": 2048,
                    "grad_checkpoint": False,
                    "clear_cache_threshold": 0,
                    "report_to": None,
                    "project_name": None,
                    "seed": 0,
                }
            )
            args = argparse.Namespace(**config)

            callback = _ProgressCallback(
                lambda pct, msg: self._update_finetune(task_id, progress=pct, message=msg),
                total_iters=iters,
            )

            try:
                self._fine_tuner(args, callback)
            except Exception as resume_err:
                err_str = str(resume_err).lower()
                if config.get("resume_adapter_file") and ("matmul" in err_str or "shape" in err_str or "dimension" in err_str):
                    self._update_finetune(task_id, progress=1, message="Resume adapter incompatible, restarting from scratch")
                    config["resume_adapter_file"] = None
                    args = argparse.Namespace(**config)
                    self._fine_tuner(args, callback)
                else:
                    raise
            adapter_weights = adapter_dir / "adapters.safetensors"
            if not adapter_weights.exists():
                adapter_weights.touch()
            self._update_finetune(
                task_id,
                status="completed",
                progress=100,
                message="Fine-tune completed",
                adapter_path=str(adapter_dir),
                finished_at=time(),
            )
            completed = self.get_finetune_task(task_id)
            if completed is not None:
                self._persist_finetune_task(completed)
        except Exception as exc:
            self._update_finetune(
                task_id,
                status="failed",
                progress=100,
                message="Fine-tune failed",
                error=str(exc),
                finished_at=time(),
            )
            failed = self.get_finetune_task(task_id)
            if failed is not None:
                self._persist_finetune_task(failed)

    def _update(self, task_id: str, **updates: str | int | float | None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)  # type: ignore[arg-type]
            task.updated_at = time()

    def _update_finetune(self, task_id: str, **updates: str | int | float | None) -> None:
        with self._lock:
            task = self._finetune_tasks.get(task_id)
            if task is None:
                return
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)  # type: ignore[arg-type]
            task.updated_at = time()
            snapshot = asdict(task)
        self._persist_finetune_task(snapshot)

    def _finetune_root(self) -> Path:
        return self.models_dir / ".finetunes"

    def _profile_runs_root(self, profile_slug: str) -> Path:
        return self._finetune_root() / "profiles" / profile_slug / "runs"

    @staticmethod
    def _normalize_profile_name(profile: str | None, *, model_id: str) -> str:
        raw = str(profile or "").strip()
        if raw:
            return raw
        base = str(model_id or "model").split("/")[-1]
        return f"{base}-default"

    @staticmethod
    def _profile_slug(profile_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", profile_name.strip().lower()).strip("-")
        return slug or "default-profile"

    @staticmethod
    def _adapter_weights_file(adapter_dir: Path) -> str | None:
        candidate = adapter_dir / "adapters.safetensors"
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)
        return None

    @staticmethod
    def _finetune_meta_path(run_root: Path) -> Path:
        return run_root / "task.json"

    def _persist_finetune_task(self, task: dict[str, str | int | float | None]) -> None:
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            return
        run_root_raw = str(task.get("run_root") or "").strip()
        run_root = Path(run_root_raw) if run_root_raw else (self._finetune_root() / task_id)
        run_root.mkdir(parents=True, exist_ok=True)
        self._write_finetune_meta(run_root, task)

    def _write_finetune_meta(self, run_root: Path, task: dict[str, str | int | float | None]) -> None:
        payload = dict(task)
        try:
            self._finetune_meta_path(run_root).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _scan_finetune_runs(self) -> list[dict[str, str | int | float | None]]:
        root = self._finetune_root()
        if not root.exists():
            return []
        out: list[dict[str, str | int | float | None]] = []
        run_dirs: list[Path] = []
        run_dirs.extend([p for p in root.glob("*") if p.is_dir() and p.name != "profiles"])
        profiles_root = root / "profiles"
        if profiles_root.exists():
            for profile_dir in profiles_root.glob("*"):
                runs_dir = profile_dir / "runs"
                if not runs_dir.exists():
                    continue
                run_dirs.extend([p for p in runs_dir.glob("*") if p.is_dir()])
        for run_root in sorted(run_dirs):
            meta = self._load_finetune_meta(run_root)
            if meta is not None:
                out.append(meta)
                continue
            recovered = self._recover_finetune_task_from_run(run_root)
            if recovered is not None:
                out.append(recovered)
        return out

    def _load_finetune_meta(self, run_root: Path) -> dict[str, str | int | float | None] | None:
        meta_path = self._finetune_meta_path(run_root)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
        except Exception:
            return None
        data.setdefault("task_id", run_root.name)
        profile_name = str(data.get("profile") or "").strip()
        if not profile_name:
            profile_name = self._profile_name_from_run(run_root)
        data.setdefault("profile", profile_name)
        data.setdefault("profile_slug", self._profile_slug(profile_name))
        data.setdefault("round", 1)
        data.setdefault("resume_adapter_file", None)
        data.setdefault("run_root", str(run_root))
        data.setdefault("merged_path", None)
        return data  # type: ignore[return-value]

    def _recover_finetune_task_from_run(self, run_root: Path) -> dict[str, str | int | float | None] | None:
        adapter_dir = run_root / "adapters"
        if not adapter_dir.exists() or not adapter_dir.is_dir():
            return None
        effective_model = ""
        model_id = ""
        fine_tune_type = "qlora"
        adapter_config = adapter_dir / "adapter_config.json"
        if adapter_config.exists():
            try:
                data = json.loads(adapter_config.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    effective_model = str(
                        data.get("model")
                        or data.get("base_model")
                        or data.get("model_path")
                        or ""
                    )
                    fine_tune_type = str(data.get("fine_tune_type") or fine_tune_type)
            except Exception:
                pass
        model_id = effective_model
        if not model_id:
            model_id = run_root.name
        profile_name = self._profile_name_from_run(run_root)
        profile_slug = self._profile_slug(profile_name)
        ts = run_root.stat().st_mtime
        return {
            "task_id": run_root.name,
            "model_id": model_id,
            "effective_model": effective_model,
            "profile": profile_name,
            "profile_slug": profile_slug,
            "round": 1,
            "status": "completed",
            "progress": 100,
            "message": "Recovered from disk",
            "fine_tune_type": fine_tune_type,
            "epochs": 0,
            "train_samples": 0,
            "started_at": ts,
            "updated_at": ts,
            "finished_at": ts,
            "adapter_path": str(adapter_dir),
            "resume_adapter_file": None,
            "run_root": str(run_root),
            "merged_path": None,
            "error": None,
        }

    @staticmethod
    def _profile_name_from_run(run_root: Path) -> str:
        parts = run_root.parts
        if "profiles" in parts:
            idx = parts.index("profiles")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "legacy-profile"

    @staticmethod
    def _model_path(model_id: str) -> Path:
        return Path(model_id.replace("/", "--"))

    @staticmethod
    def _default_fine_tuner(args: Any, callback: Any) -> None:
        from mlx_lm import lora as mlx_lora  # type: ignore

        mlx_lora.run(args, training_callback=callback)

    def _default_downloader(self, model_id: str, target_dir: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required for downloads. Install with `pip install huggingface_hub`."
            ) from exc

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        local_dir = self.models_dir / target_dir
        local_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            resume_download=True,
        )
        return local_dir

    @staticmethod
    def _fuse_adapter_into_model(
        *,
        effective_model: str,
        adapter_path: Path,
        target_path: Path,
    ) -> None:
        if not adapter_path.exists() or not adapter_path.is_dir():
            raise ValueError(f"Adapter directory not found: {adapter_path}")

        try:
            from mlx_lm import fuse as mlx_fuse  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "mlx-lm is required for merge/save. Install with `pip install 'amlx[mlx]'`."
            ) from exc

        model, tokenizer, config = mlx_fuse.load(
            effective_model,
            adapter_path=str(adapter_path),
            return_config=True,
        )
        fused_linears = [
            (name, module.fuse(dequantize=False))
            for name, module in model.named_modules()
            if hasattr(module, "fuse")
        ]
        if fused_linears:
            model.update_modules(mlx_fuse.tree_unflatten(fused_linears))

        mlx_fuse.save(
            target_path,
            effective_model,
            model,
            tokenizer,
            config,
            donate_model=False,
        )

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except Exception:
            return total
        return total

    @staticmethod
    def _estimate_repo_size_bytes(model_id: str) -> int | None:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(model_id, files_metadata=True)
            siblings = getattr(info, "siblings", None) or []
            total = 0
            for sib in siblings:
                size = getattr(sib, "size", None)
                if isinstance(size, int) and size > 0:
                    total += size
            return total if total > 0 else None
        except Exception:
            return None

    def _default_search_provider(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required for online search. Install with `pip install huggingface_hub`."
            ) from exc

        api = HfApi()
        models = api.list_models(
            author="mlx-community",
            search=query,
            sort="downloads",
            limit=limit,
        )
        out: list[dict[str, Any]] = []
        for model in models:
            model_id = str(getattr(model, "id", ""))
            if not model_id:
                continue
            disk_gb, _, _ = self._estimate_requirements(model_id)
            out.append(
                {
                    "id": model_id,
                    "label": model_id.split("/", 1)[-1],
                    "tags": "online, mlx-community",
                    "size": f"~{disk_gb:.1f} GB",
                }
            )
        return out

    @staticmethod
    def _total_ram_gb() -> float:
        if platform.system().lower() == "darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                return int(out) / (1024**3)
            except Exception:
                pass

        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            return (page_size * total_pages) / (1024**3)
        except Exception:
            return 0.0

    @staticmethod
    def _compatibility(item: CatalogItem, profile: dict[str, Any]) -> dict[str, Any]:
        min_ram = float(item.get("min_ram_gb", 8.0))
        rec_ram = float(item.get("rec_ram_gb", min_ram))
        disk = float(item.get("disk_gb", 6.0))
        requires_apple = bool(item.get("requires_apple_silicon", False))

        ram = float(profile.get("ram_gb", 0.0))
        free_disk = float(profile.get("free_disk_gb", 0.0))
        apple = bool(profile.get("apple_silicon", False))

        reasons: list[str] = []
        fit = "good"
        suitable = True

        if requires_apple and not apple:
            fit = "no"
            suitable = False
            reasons.append("Needs Apple Silicon for MLX runtime")

        if free_disk < disk:
            fit = "no"
            suitable = False
            reasons.append(f"Need ~{disk:.1f} GB free disk")

        if ram < min_ram:
            fit = "no"
            suitable = False
            reasons.append(f"Need at least {min_ram:.0f} GB RAM")
        elif ram < rec_ram and fit != "no":
            fit = "tight"
            reasons.append(f"Runs, but {rec_ram:.0f} GB RAM is recommended")
        elif fit != "no":
            fit = "good"
            reasons.append("Good fit for this machine")

        return {
            "fit": fit,
            "suitable": suitable,
            "summary": reasons[0] if reasons else "Unknown",
            "reasons": reasons,
            "required_ram_gb": min_ram,
            "recommended_ram_gb": rec_ram,
            "required_disk_gb": disk,
        }

    @staticmethod
    def _estimate_requirements(model_id: str) -> tuple[float, float, float]:
        lower = model_id.lower()
        match = re.search(r"(\d+(?:\.\d+)?)b", lower)
        params_b = float(match.group(1)) if match else 7.0

        if "2bit" in lower:
            bytes_per_param = 0.30
        elif "3bit" in lower:
            bytes_per_param = 0.40
        elif "4bit" in lower or "int4" in lower or "q4" in lower:
            bytes_per_param = 0.55
        elif "8bit" in lower or "int8" in lower:
            bytes_per_param = 1.05
        else:
            bytes_per_param = 0.70

        weights_gb = params_b * bytes_per_param
        disk_gb = max(0.5, round(weights_gb * 1.6, 1))
        min_ram_gb = max(4.0, round(weights_gb * 1.8 + 2.0, 1))
        rec_ram_gb = max(min_ram_gb + 2.0, round(weights_gb * 2.5 + 2.0, 1))
        return disk_gb, min_ram_gb, rec_ram_gb
