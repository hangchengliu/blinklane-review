from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.analysis.pipeline import SegmentAnalysis
from backend.app.analysis.types import EventCandidate
from backend.app.db import connect, init_db
from backend.app.main import INTERRUPTED_JOB_MESSAGE, app
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
        json={
            "review_status": "confirmed",
            "location": "测试路段",
            "note": "确认",
            "plate": "FIXTURE1",
        },
    )
    exported = client.post(f"/api/events/{event_id}/export")

    assert review.status_code == 200
    assert review.json()["review_status"] == "confirmed"
    assert review.json()["plate"] == "FIXTURE1"
    assert exported.status_code == 200
    assert exported.json()["download_url"].startswith("/api/exports/")


def test_reimport_rescans_and_updates_segments(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")
    client = TestClient(app)

    first = client.post("/api/import", json={"folder_path": str(tmp_path)})
    (tmp_path / "2026-05-25_18-30-02-left_repeater.mp4").write_bytes(b"fake")
    second = client.post("/api/import", json={"folder_path": str(tmp_path)})
    (tmp_path / "2026-05-25_18-30-02-left_repeater.mp4").unlink()
    third = client.post("/api/import", json={"folder_path": str(tmp_path)})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert first.json()["session"]["id"] == second.json()["session"]["id"]
    assert second.json()["reused"] is False
    assert [item["camera"] for item in second.json()["segments"]] == ["front", "left_repeater"]
    assert third.json()["reused"] is False
    assert [item["camera"] for item in third.json()["segments"]] == ["front"]


def _candidate(track_id: int) -> EventCandidate:
    return EventCandidate(
        track_id=track_id,
        vehicle_type="car",
        start_s=1.0,
        end_s=4.0,
        key_s=2.0,
        lane_change_score=0.8,
        turn_signal_score=0.0,
        confidence=0.8,
        reason_labels=["lane_change_candidate", "turn_signal_not_observed"],
    )


def test_reanalyze_replaces_events_for_segment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")
    produced = {"track_id": 1}

    def fake_analyze(*_args, **_kwargs) -> SegmentAnalysis:
        return SegmentAnalysis(events=[_candidate(produced["track_id"])], warnings=[])

    monkeypatch.setattr("backend.app.main.analyze_segment", fake_analyze)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path)})
    session_id = imported.json()["session"]["id"]

    first = client.post("/api/analyze", json={"session_id": session_id})
    produced["track_id"] = 9
    second = client.post("/api/analyze", json={"session_id": session_id})
    events = client.get("/api/events", params={"session_id": session_id})
    finished = client.get(f"/api/jobs/{second.json()['id']}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert finished.json()["status"] == "completed"
    assert events.status_code == 200
    body = events.json()
    assert len(body) == 1
    assert body[0]["track_id"] == 9


def test_asset_warnings_are_stored_on_the_job(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")

    def fake_analyze(*_args, **_kwargs) -> SegmentAnalysis:
        return SegmentAnalysis(
            events=[_candidate(3)],
            warnings=["raw_clip failed for track 3: encoder unavailable"],
        )

    monkeypatch.setattr("backend.app.main.analyze_segment", fake_analyze)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path)})
    session_id = imported.json()["session"]["id"]
    created = client.post("/api/analyze", json={"session_id": session_id})
    job = client.get(f"/api/jobs/{created.json()['id']}")
    events = client.get("/api/events", params={"session_id": session_id})

    assert created.status_code == 200
    assert job.json()["status"] == "completed"
    assert "raw_clip failed for track 3" in job.json()["message"]
    assert events.json()[0]["raw_clip_url"] is None


def test_startup_marks_leftover_jobs_failed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    session_id = new_id()
    job_id = new_id()
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, folder_path, created_at) VALUES (?, ?, ?)",
            (session_id, str(tmp_path), now),
        )
        conn.execute(
            """
            INSERT INTO analysis_jobs (
              id, session_id, status, progress, message, model_name,
              sample_rate_fps, created_at, updated_at
            ) VALUES (?, ?, 'running', 0.4, 'Starting analysis', 'yolo26s.pt', 5, ?, ?)
            """,
            (job_id, session_id, now, now),
        )

    with TestClient(app):
        pass

    with connect() as conn:
        row = conn.execute(
            "SELECT status, message FROM analysis_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

    assert row["status"] == "failed"
    assert row["message"] == INTERRUPTED_JOB_MESSAGE
