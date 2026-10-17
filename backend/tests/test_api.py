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
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    client = TestClient(app)

    response = client.post("/api/import", json={"folder_path": str(tmp_path)})

    assert response.status_code == 400


def test_import_folder_and_reuse_existing_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
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


def _seed_session_with_event(tmp_path: Path) -> tuple[str, str, str]:
    """Insert one session, one front segment, and one pending event."""

    data_dir = tmp_path / "data"
    raw = data_dir / "clips" / "raw.mp4"
    key = data_dir / "keys" / "key.jpg"
    raw.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"raw")
    key.write_bytes(b"jpg")
    session_id, segment_id, event_id = new_id(), new_id(), new_id()
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
    return session_id, segment_id, event_id


def test_media_source_serves_only_registered_mp4(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    folder = tmp_path / "folder"
    folder.mkdir()
    clip = folder / "2026-05-25_18-30-02-front.mp4"
    clip.write_bytes(b"fake")
    stray = tmp_path / "stray.mp4"
    stray.write_bytes(b"fake")
    not_mp4 = tmp_path / "notes.txt"
    not_mp4.write_text("nope", encoding="utf-8")
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(folder)})
    assert imported.status_code == 200

    registered = client.get("/api/media/source", params={"path": str(clip.resolve())})
    unregistered = client.get("/api/media/source", params={"path": str(stray.resolve())})
    wrong_type = client.get("/api/media/source", params={"path": str(not_mp4.resolve())})

    assert registered.status_code == 200
    assert unregistered.status_code == 403
    assert wrong_type.status_code == 403


def test_media_path_traversal_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"fake")
    client = TestClient(app)

    response = client.get("/api/media/%2E%2E/secret.mp4")

    assert response.status_code == 403


def test_analyze_rejects_invalid_model_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path)})
    session_id = imported.json()["session"]["id"]

    invalid = [
        "../evil.pt",
        "weights/yolo26s.pt",
        "weights\\yolo26s.pt",
        "https://example.test/yolo26s.pt",
        "yolo26s",
        "yolo26s.pt.bak",
    ]
    for model_name in invalid:
        response = client.post(
            "/api/analyze",
            json={"session_id": session_id, "model_name": model_name},
        )
        assert response.status_code == 422, model_name


def test_analyze_accepts_plain_model_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")

    def fake_analyze(*_args, **_kwargs) -> SegmentAnalysis:
        return SegmentAnalysis(events=[], warnings=[])

    monkeypatch.setattr("backend.app.main.analyze_segment", fake_analyze)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path)})
    session_id = imported.json()["session"]["id"]

    response = client.post(
        "/api/analyze",
        json={"session_id": session_id, "model_name": "yolo26s.pt"},
    )

    assert response.status_code == 200
    assert response.json()["model_name"] == "yolo26s.pt"


def test_analyze_returns_409_when_job_already_active(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    session_id = new_id()
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
            ) VALUES (?, ?, 'queued', 0, 'Queued', 'yolo26s.pt', 5, ?, ?)
            """,
            (new_id(), session_id, now, now),
        )
    client = TestClient(app)

    response = client.post("/api/analyze", json={"session_id": session_id})

    assert response.status_code == 409


def test_reanalyze_preserves_confirmed_events_and_exports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")

    def fake_analyze(*_args, **_kwargs) -> SegmentAnalysis:
        return SegmentAnalysis(events=[_candidate(1)], warnings=[])

    monkeypatch.setattr("backend.app.main.analyze_segment", fake_analyze)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path)})
    session_id = imported.json()["session"]["id"]

    client.post("/api/analyze", json={"session_id": session_id})
    event = client.get("/api/events", params={"session_id": session_id}).json()[0]
    review = client.patch(
        f"/api/events/{event['id']}/review",
        json={"review_status": "confirmed", "location": "路段", "note": "", "plate": "K1"},
    )
    assert review.status_code == 200
    assert review.json()["export"]["zip_path"]

    second = client.post("/api/analyze", json={"session_id": session_id})
    assert second.status_code == 200
    events = client.get("/api/events", params={"session_id": session_id}).json()

    # The confirmed event survives; the re-analysis adds a fresh pending copy.
    assert len(events) == 2
    by_status = {item["review_status"]: item for item in events}
    assert by_status["confirmed"]["id"] == event["id"]
    assert by_status["pending"]["id"] != event["id"]
    with connect() as conn:
        reviews = conn.execute(
            "SELECT COUNT(*) AS count FROM event_reviews WHERE event_id = ?",
            (event["id"],),
        ).fetchone()["count"]
        exports = conn.execute(
            "SELECT COUNT(*) AS count FROM exports WHERE event_id = ?",
            (event["id"],),
        ).fetchone()["count"]
    assert reviews == 1
    assert exports == 1


def test_rescan_keeps_segments_that_have_events(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    front = tmp_path / "2026-05-25_18-30-02-front.mp4"
    back = tmp_path / "2026-05-25_18-30-02-back.mp4"
    front.write_bytes(b"fake")
    back.write_bytes(b"fake")

    def fake_analyze(*_args, **_kwargs) -> SegmentAnalysis:
        return SegmentAnalysis(events=[_candidate(1)], warnings=[])

    monkeypatch.setattr("backend.app.main.analyze_segment", fake_analyze)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path)})
    session_id = imported.json()["session"]["id"]
    client.post("/api/analyze", json={"session_id": session_id})

    front.unlink()
    rescanned = client.post("/api/import", json={"folder_path": str(tmp_path)})

    assert rescanned.status_code == 200
    cameras = [item["camera"] for item in rescanned.json()["segments"]]
    # The front segment is gone from disk but stays because it has events.
    assert sorted(cameras) == ["back", "front"]
    events = client.get("/api/events", params={"session_id": session_id}).json()
    assert len(events) == 1


def test_rescan_with_zero_clips_keeps_existing_segments(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    clip = tmp_path / "2026-05-25_18-30-02-front.mp4"
    clip.write_bytes(b"fake")
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path)})
    assert len(imported.json()["segments"]) == 1

    clip.unlink()
    rescanned = client.post("/api/import", json={"folder_path": str(tmp_path)})

    assert rescanned.status_code == 200
    assert rescanned.json()["reused"] is True
    assert len(rescanned.json()["segments"]) == 1


def test_review_survives_export_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    init_db()
    _, _, event_id = _seed_session_with_event(tmp_path)

    def explode(*_args, **_kwargs) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr("backend.app.main.export_evidence_package", explode)
    client = TestClient(app)

    review = client.patch(
        f"/api/events/{event_id}/review",
        json={"review_status": "confirmed", "location": "路段", "note": "", "plate": "F9"},
    )

    assert review.status_code == 200
    body = review.json()
    assert body["review_status"] == "confirmed"
    assert "export" not in body
    assert "disk full" in body["export_error"]
    with connect() as conn:
        row = conn.execute(
            "SELECT review_status FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        reviews = conn.execute(
            "SELECT COUNT(*) AS count FROM event_reviews WHERE event_id = ?",
            (event_id,),
        ).fetchone()["count"]
    assert row["review_status"] == "confirmed"
    assert reviews == 1
