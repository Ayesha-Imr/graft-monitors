"""Disk cache for monitor scores, keyed by hash(model, question, prompt, completion).

SQLite in WAL mode so several processes on one Pod (trainer, evals) can share it.
Rescoring for audits reads this cache, which is what keeps hosted-API drift out
of reanalysis.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


def cache_key(model: str, question: str, prompt: str, completion: str) -> str:
    h = hashlib.sha256()
    for part in (model, question, prompt, completion):
        b = part.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


class ScoreCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL,"
                      " model TEXT, meta TEXT, created REAL)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def get_many(self, keys: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        conn = self._conn()
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = "SELECT key, score FROM scores WHERE key IN (%s)" % ",".join("?" * len(chunk))
            out.update(dict(conn.execute(q, chunk).fetchall()))
        return out

    def put_many(self, rows: list[tuple[str, float, str, dict]]) -> None:
        now = time.time()
        conn = self._conn()
        conn.execute("BEGIN")
        conn.executemany("INSERT OR IGNORE INTO scores VALUES (?, ?, ?, ?, ?)",
                         [(k, float(s), m, json.dumps(meta), now) for k, s, m, meta in rows])
        conn.execute("COMMIT")

    def meta(self, key: str) -> dict | None:
        row = self._conn().execute("SELECT meta FROM scores WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
