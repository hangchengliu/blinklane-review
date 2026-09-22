from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.adapters import Evidence, LocalExportAdapter, SubmissionResult
from backend.app.db import connect, init_db
from backend.app.main import app
from backend.app.tesla import new_id


def test_local_adapter_rejects_unconfirmed_events(tmp_path: Path) -> None:
    evidence = Evidence(
        event_id="evt",
        clip=None,
        screenshot=None,
        absolute_time=None,
        place="",
        plate="",
        confirm_status="pending",
        package_dir=str(tmp_path),
        zip_path=str(tmp_path / "missing.zip"),
    )

    result = LocalExportAdapter().submit(evidence)

    assert result.ok is False
    assert result.adapter == "local_export"
    assert "confirmed" in result.message


def test_export_submits_confirmed_evidence_through_the_adapter(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(data_dir))
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
            (segment_id, session_id, "2026-05-25T18:30:02", str(tmp_path / "front.mp4"), now),
        )
        conn.execute(
            """
            INSERT INTO events (
              id, session_id, segment_id, camera, track_id, vehicle_type,
              start_s, end_s, key_s, lane_change_score, turn_signal_score,
              confidence, reason_labels, review_status, location, note, plate,
              raw_clip_path, key_frame_path, created_at
            ) VALUES (
              ?, ?, ?, 'front', 1, 'car', 1, 5, 3, 0.9, 0, 0.9, ?, 'confirmed', ?, ?, ?, ?, ?, ?
            )
            """,
            (
                event_id,
                session_id,
                segment_id,
                json.dumps(["lane_change_candidate"]),
                "测试路段",
                "确认",
                "FIXTURE1",
                str(raw),
                str(key),
                now,
            ),
        )

    seen: list[Evidence] = []

    class RecordingAdapter:
        name = "recording"

        def submit(self, evidence: Evidence) -> SubmissionResult:
            seen.append(evidence)
            return SubmissionResult(
                ok=True,
                adapter=self.name,
                message="recorded",
                location=evidence.zip_path,
            )

    monkeypatch.setattr("backend.app.main.get_upload_adapter", lambda: RecordingAdapter())
    client = TestClient(app)
    exported = client.post(f"/api/events/{event_id}/export")

    assert exported.status_code == 200
    assert exported.json()["adapter"] == "recording"
    assert len(seen) == 1
    evidence = seen[0]
    assert evidence.confirm_status == "confirmed"
    assert evidence.place == "测试路段"
    assert evidence.plate == "FIXTURE1"
    assert evidence.clip == str(raw)
    assert evidence.screenshot == str(key)
    assert evidence.absolute_time == "2026-05-25 18:30:03 至 2026-05-25 18:30:07"
