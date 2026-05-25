from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.db import connect, init_db
from backend.app.main import app
from backend.app.tesla import new_id


def test_import_empty_folder_returns_400(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VCY_DATA_DIR", str(tmp_path / "data"))
    init_db()
    client = TestClient(app)

    response = client.post("/api/import", json={"folder_path": str(tmp_path)})

    assert response.status_code == 400


def test_import_folder_and_reuse_existing_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VCY_DATA_DIR", str(tmp_path / "data"))
    init_db()
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")
    client = TestClient(app)

    first = client.post("/api/import", json={"folder_path": str(tmp_path)})
    second = client.post("/api/import", json={"folder_path": str(tmp_path)})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session"]["id"] == second.json()["session"]["id"]
    assert second.json()["reused"] is True


def test_review_and_export_confirmed_event(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VCY_DATA_DIR", str(data_dir))
    init_db()
    session_id = new_id()
    segment_id = new_id()
    event_id = new_id()
    raw = data_dir / "clips" / "raw.mp4"
    key = data_dir / "keys" / "key.jpg"
    raw.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"raw")
    key.write_bytes(b"jpg")
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, folder_path, created_at) VALUES (?, ?, ?)",
            (session_id, str(tmp_path), now),
        )
        conn.execute(
            """
            INSERT INTO video_segments (id, session_id, camera, starts_at, path, created_at)
            VALUES (?, ?, 'front', ?, ?, ?)
            """,
            (segment_id, session_id, now, str(tmp_path / "front.mp4"), now),
        )
        conn.execute(
            """
            INSERT INTO events (
              id, session_id, segment_id, camera, track_id, vehicle_type,
              start_s, end_s, key_s, lane_change_score, turn_signal_score,
              confidence, reason_labels, raw_clip_path, key_frame_path, created_at
            ) VALUES (?, ?, ?, 'front', 1, 'car', 1, 5, 3, 0.9, 0, 0.9, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                segment_id,
                json.dumps(["lane_change_candidate"], ensure_ascii=False),
                str(raw),
                str(key),
                now,
            ),
        )

    client = TestClient(app)
    review = client.patch(
        f"/api/events/{event_id}/review",
        json={"review_status": "confirmed", "location": "测试路段", "note": "确认"},
    )
    exported = client.post(f"/api/events/{event_id}/export")

    assert review.status_code == 200
    assert review.json()["review_status"] == "confirmed"
    assert exported.status_code == 200
    assert exported.json()["download_url"].startswith("/api/exports/")
