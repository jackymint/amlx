from __future__ import annotations

import sqlite3
from pathlib import Path


class DiskCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch())
                )
                """
            )

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM cache_entries WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE cache_entries SET hits = hits + 1, updated_at = unixepoch() WHERE key = ?",
                (key,),
            )
            return str(row[0])

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache_entries")

    def put(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries(key, value, hits, updated_at)
                VALUES (?, ?, 0, unixepoch())
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = unixepoch()
                """,
                (key, value),
            )
