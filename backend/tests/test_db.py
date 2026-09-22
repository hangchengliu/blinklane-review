from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.db import SQLITE_TIMEOUT_SECONDS, connect


def test_connection_uses_timeout_and_closes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))

    with connect() as conn:
        held = conn
        timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert timeout_ms == int(SQLITE_TIMEOUT_SECONDS * 1000)
        assert str(journal).lower() == "wal"

    with pytest.raises(sqlite3.ProgrammingError):
        held.execute("SELECT 1")
