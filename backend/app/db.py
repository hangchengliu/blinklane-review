"""SQLite access.

Connections set a busy timeout and are closed by the context manager.
``with connect()`` therefore commits on success, rolls back on error, and
always closes. Callers that run long work (analysis) must not hold a
connection across that work: open one for each short read or write. WAL
mode lets those short writers run while API requests read.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import ensure_data_dirs, get_settings

SQLITE_TIMEOUT_SECONDS = 30.0

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  folder_path TEXT NOT NULL UNIQUE,
  source_kind TEXT NOT NULL DEFAULT 'tesla_dashcam',
  status TEXT NOT NULL DEFAULT 'imported',
  created_at TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS video_segments (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  camera TEXT NOT NULL,
  starts_at TEXT NOT NULL,
  path TEXT NOT NULL,
  duration_s REAL,
  fps REAL,
  width INTEGER,
  height INTEGER,
  created_at TEXT NOT NULL,
  UNIQUE(session_id, path)
);

CREATE TABLE IF NOT EXISTS analysis_jobs (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0,
  message TEXT NOT NULL DEFAULT '',
  model_name TEXT NOT NULL,
  sample_rate_fps REAL NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  segment_id TEXT NOT NULL REFERENCES video_segments(id) ON DELETE CASCADE,
  camera TEXT NOT NULL,
  track_id INTEGER NOT NULL,
  vehicle_type TEXT NOT NULL,
  start_s REAL NOT NULL,
  end_s REAL NOT NULL,
  key_s REAL NOT NULL,
  lane_change_score REAL NOT NULL,
  turn_signal_score REAL NOT NULL,
  confidence REAL NOT NULL,
  reason_labels TEXT NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'pending',
  location TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  raw_clip_path TEXT,
  annotated_clip_path TEXT,
  key_frame_path TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_reviews (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  location TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exports (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  zip_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    ensure_data_dirs(settings)
    db_path = db_path or settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=SQLITE_TIMEOUT_SECONDS,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={int(SQLITE_TIMEOUT_SECONDS * 1000)}")
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    if "reason_labels" in item and isinstance(item["reason_labels"], str):
        try:
            item["reason_labels"] = json.loads(item["reason_labels"])
        except json.JSONDecodeError:
            item["reason_labels"] = []
    return item


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [row_to_dict(row) or {} for row in rows]
