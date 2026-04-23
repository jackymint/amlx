from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from amlx.adapters.base import GenerationResult, ModelAdapter
from amlx.scheduler import BatchScheduler


class SpyAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, *, model: str, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        del model, max_tokens, temperature
        return GenerationResult(text=prompt.upper(), prompt_tokens=1, completion_tokens=1)

    def generate_batch(
        self,
        *,
        model: str,
        prompts: list[str],
        max_tokens: int,
        temperature: float,
    ) -> list[GenerationResult]:
        del model, max_tokens, temperature
        self.batch_sizes.append(len(prompts))
        time.sleep(0.02)
        return [GenerationResult(text=p.upper(), prompt_tokens=1, completion_tokens=1) for p in prompts]


def test_scheduler_batches_parallel_requests() -> None:
    adapter = SpyAdapter()
    scheduler = BatchScheduler(adapter=adapter, max_batch_size=8, max_wait_ms=60)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future1 = pool.submit(
                scheduler.submit,
                model="m",
                prompt="hello",
                max_tokens=16,
                temperature=0.2,
            )
            future2 = pool.submit(
                scheduler.submit,
                model="m",
                prompt="world",
                max_tokens=16,
                temperature=0.2,
            )
            res1 = future1.result(timeout=2)
            res2 = future2.result(timeout=2)

        assert res1.text == "HELLO"
        assert res2.text == "WORLD"
        assert max(adapter.batch_sizes) >= 2
    finally:
        scheduler.close()
