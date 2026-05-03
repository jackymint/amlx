from __future__ import annotations

from .shared import *

class ModelManagerCatalogOpsMixin:
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

    def model_arch_info(self, model_id: str) -> dict[str, str | int | None]:
        for item in self.installed_models():
            if item.get("model_id") != model_id:
                continue
            config_path = Path(item["path"]) / "config.json"
            if not config_path.exists():
                break
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                break
            return {
                "model_type": cfg.get("model_type"),
                "num_hidden_layers": cfg.get("num_hidden_layers"),
                "hidden_size": cfg.get("hidden_size"),
                "num_attention_heads": cfg.get("num_attention_heads"),
                "max_position_embeddings": cfg.get("max_position_embeddings"),
            }
        return {}

    def receive_imported_model(self, folder_name: str, model_id: str, file_pairs: list[tuple[str, object]]) -> dict[str, str]:
        dest = self.models_dir / folder_name
        if dest.exists():
            raise ValueError(f"Model folder already exists: {folder_name}")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            for rel_path, file_obj in file_pairs:
                target = dest / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "wb") as f:
                    shutil.copyfileobj(file_obj, f)
            marker = dest / "amlx_model.json"
            marker.write_text(
                json.dumps({"model_id": model_id, "imported_at": int(time())}),
                encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        return {"model_id": model_id, "path": str(dest)}

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

        try:
            (resolved / "amlx_model.json").unlink(missing_ok=True)
        except Exception:
            pass
        threading.Thread(target=shutil.rmtree, args=(resolved,), kwargs={"ignore_errors": True}, daemon=True).start()
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

    def cancel_download(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status not in {"queued", "downloading"}:
                return False
            task.cancelled = True
            task.status = "cancelled"
            task.message = "Cancelled"
            task.updated_at = time()
            proc = self._download_procs.get(task_id)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        return True

    def enqueue_quantize(
        self,
        *,
        model_id: str,
        effective_model: str,
        output_path: str,
        q_bits: int = 4,
        q_group_size: int = 64,
    ) -> dict[str, object]:
        task_id = str(uuid.uuid4())
        task = QuantizeTask(
            task_id=task_id,
            model_id=model_id,
            effective_model=effective_model,
            output_path=output_path,
            q_bits=q_bits,
            q_group_size=q_group_size,
            status="queued",
            progress=0,
            message="Queued",
            started_at=time(),
            updated_at=time(),
        )
        with self._lock:
            self._quantize_tasks[task_id] = task
        thread = threading.Thread(target=self._run_quantize, args=(task_id,), daemon=True)
        thread.start()
        return asdict(task)

    def cancel_quantize(self, task_id: str) -> bool:
        with self._lock:
            task = self._quantize_tasks.get(task_id)
            if task is None or task.status not in {"queued", "running"}:
                return False
            task.cancelled = True
            task.status = "cancelled"
            task.message = "Cancelled"
            task.updated_at = time()
            proc = self._quantize_procs.get(task_id)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        print(f"[amlx] quantize cancelled: {task_id}", flush=True)
        return True

    def list_quantize_tasks(self) -> list[dict[str, object]]:
        with self._lock:
            tasks = sorted(self._quantize_tasks.values(), key=lambda x: x.started_at, reverse=True)
            return [asdict(t) for t in tasks]

    def _update_quantize(self, task_id: str, **kwargs: object) -> None:
        with self._lock:
            task = self._quantize_tasks.get(task_id)
            if task is None:
                return
            for k, v in kwargs.items():
                if hasattr(task, k):
                    object.__setattr__(task, k, v)
            task.updated_at = time()
