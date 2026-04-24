from __future__ import annotations

from .shared import *

class ModelManagerQuantizeTasksMixin:
    def _run_quantize(self, task_id: str) -> None:
        with self._lock:
            task = self._quantize_tasks.get(task_id)
        if task is None:
            return

        model_id = task.model_id
        effective_model = task.effective_model
        output_path = task.output_path
        q_bits = task.q_bits
        q_group_size = task.q_group_size

        self._update_quantize(task_id, status="running", progress=1, message="Starting quantization")
        print(f"[amlx] quantize start: {model_id} → {q_bits}bit → {output_path}", flush=True)

        try:
            from mlx_lm import convert as mlx_convert

            error_holder: list[Exception] = []
            done = threading.Event()

            def run_convert() -> None:
                try:
                    mlx_convert(
                        hf_path=effective_model,
                        mlx_path=output_path,
                        quantize=True,
                        q_bits=q_bits,
                        q_group_size=q_group_size,
                    )
                except Exception as exc:
                    error_holder.append(exc)
                finally:
                    done.set()

            worker = threading.Thread(target=run_convert, daemon=True)
            worker.start()

            progress = 1
            while not done.wait(timeout=2.0):
                with self._lock:
                    if self._quantize_tasks.get(task_id) and self._quantize_tasks[task_id].cancelled:
                        return
                progress = min(95, progress + 2)
                self._update_quantize(task_id, progress=progress, message="Quantizing layers...")

            worker.join(timeout=0.1)
            if error_holder:
                raise error_holder[0]

            self._update_quantize(
                task_id,
                status="completed",
                progress=100,
                message="Done",
                finished_at=time(),
            )
            print(f"[amlx] quantize done: {output_path}", flush=True)
        except Exception as exc:
            self._update_quantize(
                task_id,
                status="failed",
                progress=0,
                message="Failed",
                error=str(exc),
                finished_at=time(),
            )
            print(f"[amlx] quantize error: {exc}", flush=True)

    def list_tasks(self) -> list[dict[str, str | int | float | None]]:
        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda x: x.started_at, reverse=True)
            return [asdict(t) for t in tasks if t.status not in {"completed", "cancelled"}]

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

    def get_task(self, task_id: str) -> dict[str, str | int | float | None] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return asdict(task)
