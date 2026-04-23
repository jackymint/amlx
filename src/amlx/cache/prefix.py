from __future__ import annotations

import hashlib
from dataclasses import dataclass

from amlx.cache.blocks import PagedBlockStore
from amlx.cache.disk import DiskCache
from amlx.cache.memory import LRUCache


@dataclass(slots=True)
class PrefixCacheStats:
    memory_hits: int = 0
    disk_hits: int = 0
    block_hits: int = 0
    misses: int = 0
    block_writes: int = 0


class PrefixCache:
    def __init__(
        self,
        memory_cache: LRUCache,
        disk_cache: DiskCache,
        block_store: PagedBlockStore | None = None,
    ) -> None:
        self.memory = memory_cache
        self.disk = disk_cache
        self.blocks = block_store
        self.stats = PrefixCacheStats()

    @staticmethod
    def build_key(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        material = f"{model}|{temperature:.3f}|{max_tokens}|{prompt}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def get(self, key: str) -> str | None:
        value = self.memory.get(key)
        if value is not None:
            self.stats.memory_hits += 1
            return value

        value = self.disk.get(key)
        if value is not None:
            self.stats.disk_hits += 1
            self.memory.put(key, value)
            return value

        if self.blocks is not None:
            value = self.blocks.get(cache_key=key)
            if value is not None:
                self.stats.block_hits += 1
                self.memory.put(key, value)
                self.disk.put(key, value)
                return value

        self.stats.misses += 1
        return None

    def clear(self) -> None:
        self.memory.clear()
        self.disk.clear()
        self.stats = PrefixCacheStats()

    def put(self, key: str, value: str) -> None:
        self.memory.put(key, value)
        self.disk.put(key, value)
        if self.blocks is not None:
            self.stats.block_writes += self.blocks.put(cache_key=key, value=value)
