import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).resolve().parents[3] / "benchmarks.sqlite3"))

_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        with _lock:
            if _connection is None:
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
                _connection.row_factory = sqlite3.Row
                _connection.execute("PRAGMA journal_mode=WAL")
    return _connection


@contextmanager
def get_session() -> Iterator[sqlite3.Connection]:
    """A SQLite connection is not safe for concurrent writers across threads
    without external serialization, so every write/read in this app goes
    through this single lock-guarded connection rather than opening one per
    request."""
    conn = _get_connection()
    with _lock:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
