from __future__ import annotations

from .shared import *

class ModelManagerQuantizeTasksMixin:
    def _run_quantize(self, task_id: str) -> None:
        import sys

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
            script = (
                "from mlx_lm import convert; "
                f"convert(hf_path={effective_model!r}, mlx_path={output_path!r}, "
                f"quantize=True, q_bits={q_bits}, q_group_size={q_group_size})"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with self._lock:
                self._quantize_procs[task_id] = proc

            progress = 1
            while proc.poll() is None:
                with self._lock:
                    if self._quantize_tasks.get(task_id) and self._quantize_tasks[task_id].cancelled:
                        proc.kill()
                        proc.wait()
                        self._quantize_procs.pop(task_id, None)
                        return
                progress = min(95, progress + 2)
                self._update_quantize(task_id, progress=progress, message="Quantizing layers...")
                sleep(2.0)

            with self._lock:
                self._quantize_procs.pop(task_id, None)

            stdout, _ = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Quantize process failed (exit {proc.returncode})")

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
