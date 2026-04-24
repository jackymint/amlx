from __future__ import annotations

from .shared import *

class ModelManagerEnqueueMixin:
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
        learning_rate: float = 1e-5,
        lora_rank: int = 8,
        lora_layers: int = 16,
        max_seq_length: int = 2048,
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
            learning_rate=float(learning_rate),
            lora_rank=int(lora_rank),
            lora_layers=int(lora_layers),
            max_seq_length=int(max_seq_length),
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
