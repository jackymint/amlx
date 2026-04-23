from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from amlx.adapters.base import GenerationResult, ModelAdapter


@dataclass(slots=True)
class SchedulerStats:
    enqueued: int = 0
    processed: int = 0
    batch_runs: int = 0
    total_batch_items: int = 0
    throttle_sleep_ms: int = 0


@dataclass(slots=True)
class _Task:
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    event: threading.Event
    result: GenerationResult | None = None
    error: Exception | None = None


class BatchScheduler:
    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        max_batch_size: int = 8,
        max_wait_ms: int = 20,
        gpu_limit_percent: int = 100,
    ) -> None:
        self.adapter = adapter
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(1, max_wait_ms)
        self.stats = SchedulerStats()
        self._gpu_limit_percent = self._clamp_gpu_limit(gpu_limit_percent)
        self._settings_lock = threading.Lock()

        self._queue: queue.Queue[_Task | None] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="amlx-batcher", daemon=True)
        self._worker.start()

    @staticmethod
    def _clamp_gpu_limit(value: int) -> int:
        return max(20, min(100, int(value)))

    def set_gpu_limit_percent(self, value: int) -> int:
        with self._settings_lock:
            self._gpu_limit_percent = self._clamp_gpu_limit(value)
            return self._gpu_limit_percent

    def gpu_limit_percent(self) -> int:
        with self._settings_lock:
            return self._gpu_limit_percent

    def submit(self, *, model: str, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        task = _Task(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            event=threading.Event(),
        )
        self.stats.enqueued += 1
        self._queue.put(task)
        task.event.wait()

        if task.error is not None:
            raise task.error
        assert task.result is not None
        return task.result

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            first = self._queue.get()
            if first is None:
                break

            batch: list[_Task] = [first]
            deadline = time.monotonic() + (self.max_wait_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                batch.append(item)

            self._execute_batch(batch)

    def _execute_batch(self, batch: list[_Task]) -> None:
        batch_started = time.monotonic()
        self.stats.batch_runs += 1
        self.stats.total_batch_items += len(batch)

        grouped: dict[tuple[str, int, float], list[tuple[int, _Task]]] = {}
        for idx, task in enumerate(batch):
            key = (task.model, task.max_tokens, task.temperature)
            grouped.setdefault(key, []).append((idx, task))

        outputs: list[GenerationResult | None] = [None] * len(batch)

        try:
            for (model, max_tokens, temperature), items in grouped.items():
                prompts = [task.prompt for _, task in items]
                results = self.adapter.generate_batch(
                    model=model,
                    prompts=prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if len(results) != len(items):
                    raise RuntimeError("Adapter returned unexpected batch size")
                for (batch_index, _), result in zip(items, results, strict=True):
                    outputs[batch_index] = result

            for task, result in zip(batch, outputs, strict=True):
                if result is None:
                    task.error = RuntimeError("Missing generation result")
                else:
                    task.result = result
                    self.stats.processed += 1
                task.event.set()

        except Exception as exc:
            for task in batch:
                task.error = exc
                task.event.set()
        finally:
            self._apply_gpu_throttle(batch_started)

    def _apply_gpu_throttle(self, batch_started: float) -> None:
        limit = self.gpu_limit_percent()
        if limit >= 100:
            return
        elapsed = max(0.0, time.monotonic() - batch_started)
        if elapsed <= 0.0:
            return
        cooldown = elapsed * ((100 - limit) / limit)
        if cooldown <= 0.0:
            return
        self.stats.throttle_sleep_ms += int(cooldown * 1000)
        time.sleep(cooldown)
