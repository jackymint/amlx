from __future__ import annotations

from .shared import *

class ModelManagerFineTuneLookupMixin:
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
