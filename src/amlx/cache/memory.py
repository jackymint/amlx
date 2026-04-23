from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class MemoryEntry:
    key: str
    value: str


class LRUCache:
    def __init__(self, capacity: int = 512) -> None:
        self.capacity = max(1, capacity)
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return entry.value

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = MemoryEntry(key=key, value=value)
            return

        self._store[key] = MemoryEntry(key=key, value=value)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
