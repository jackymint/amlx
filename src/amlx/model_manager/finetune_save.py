from __future__ import annotations

from .shared import *

class ModelManagerFineTuneSaveMixin:
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
