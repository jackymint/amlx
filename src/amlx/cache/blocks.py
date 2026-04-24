from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class PagedBlockStore:
    """Stores cache values as fixed-size text blocks on disk.

    This mimics paged cache behavior used by KV systems and keeps data
    recoverable across process restarts.
    """

    def __init__(self, *, root_dir: Path, index_db: Path, block_chars: int = 4096) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_db = index_db
        self.index_db.parent.mkdir(parents=True, exist_ok=True)
        self.block_chars = max(1, block_chars)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.index_db)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS block_refs (
                    block_id TEXT PRIMARY KEY,
                    rel_path TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch())
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manifests (
                    cache_key TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    block_id TEXT NOT NULL,
                    PRIMARY KEY(cache_key, seq)
                )
                """
            )

    def _split(self, value: str) -> list[str]:
        if not value:
            return [""]
        return [value[i : i + self.block_chars] for i in range(0, len(value), self.block_chars)]

    def _block_path(self, block_id: str) -> tuple[Path, str]:
        rel = f"{block_id[:2]}/{block_id[2:4]}/{block_id}.blk"
        return self.root_dir / rel, rel

    def put(self, *, cache_key: str, value: str) -> int:
        blocks = self._split(value)
        block_ids: list[str] = []

        with self._connect() as conn:
            conn.execute("DELETE FROM manifests WHERE cache_key = ?", (cache_key,))
            for seq, block in enumerate(blocks):
                payload = block.encode("utf-8")
                block_id = hashlib.sha256(payload).hexdigest()
                block_ids.append(block_id)
                path, rel = self._block_path(block_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(payload)

                conn.execute(
                    """
                    INSERT OR IGNORE INTO block_refs(block_id, rel_path, size)
                    VALUES (?, ?, ?)
                    """,
                    (block_id, rel, len(payload)),
                )
                conn.execute(
                    """
                    INSERT INTO manifests(cache_key, seq, block_id)
                    VALUES (?, ?, ?)
                    """,
                    (cache_key, seq, block_id),
                )

        return len(block_ids)

    def get(self, *, cache_key: str) -> str | None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.block_id, b.rel_path
                FROM manifests m
                JOIN block_refs b ON b.block_id = m.block_id
                WHERE m.cache_key = ?
                ORDER BY m.seq ASC
                """,
                (cache_key,),
            ).fetchall()

        if not rows:
            return None

        out: list[str] = []
        for _, rel_path in rows:
            path = self.root_dir / str(rel_path)
            if not path.exists():
                return None
            out.append(path.read_text(encoding="utf-8"))

        return "".join(out)

    def clear(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT rel_path FROM block_refs").fetchall()
            conn.execute("DELETE FROM manifests")
            conn.execute("DELETE FROM block_refs")

        for (rel_path,) in rows:
            try:
                path = self.root_dir / str(rel_path)
                if path.exists():
                    path.unlink()
            except Exception:
                continue
