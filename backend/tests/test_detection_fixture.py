"""Detection changes are checked against the synthetic fixture, not real footage."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.analysis.pipeline import analyze_segment
from backend.app.analysis.trajectory import LATERAL_SHIFT_THRESHOLD
from backend.app.main import app
from backend.tests.synthetic import (
    install_fake_detector,
    write_blinking_repeater,
    write_dark_clip,
    write_lane_crossing_clip,
    write_moving_car_clip,
)


def test_fixture_event_stays_above_the_shift_threshold(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    folder = tmp_path / "plain"
    front = folder / "2026-05-25_18-30-02-front.mp4"
    write_moving_car_clip(front)
    write_dark_clip(folder / "2026-05-25_18-30-02-left_repeater.mp4")
    write_dark_clip(folder / "2026-05-25_18-30-02-right_repeater.mp4")
    install_fake_detector(monkeypatch)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(folder)})
    session_id = imported.json()["session"]["id"]
    created = client.post("/api/analyze", json={"session_id": session_id})
    job = client.get(f"/api/jobs/{created.json()['id']}")
    events = client.get("/api/events", params={"session_id": session_id}).json()

    assert job.json()["status"] == "completed", job.json()["message"]
    assert LATERAL_SHIFT_THRESHOLD == 0.12
    assert len(events) == 1
    assert events[0]["lane_change_score"] > LATERAL_SHIFT_THRESHOLD
    assert "turn_signal_not_observed" in events[0]["reason_labels"]


def test_time_aligned_repeater_blink_suppresses_fixture_event(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    folder = tmp_path / "blink"
    front = folder / "2026-05-25_18-30-02-front.mp4"
    write_moving_car_clip(front)
    write_blinking_repeater(folder / "2026-05-25_18-30-02-left_repeater.mp4")
    write_dark_clip(folder / "2026-05-25_18-30-02-right_repeater.mp4")
    install_fake_detector(monkeypatch)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(folder)})
    session_id = imported.json()["session"]["id"]
    created = client.post("/api/analyze", json={"session_id": session_id})
    job = client.get(f"/api/jobs/{created.json()['id']}")
    events = client.get("/api/events", params={"session_id": session_id}).json()

    assert job.json()["status"] == "completed", job.json()["message"]
    assert events == []


def test_visible_lane_lines_change_the_fixture_score(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    front = tmp_path / "2026-05-25_18-30-02-front.mp4"
    write_lane_crossing_clip(front)
    install_fake_detector(monkeypatch, x0=250, step_px=6)

    with_lanes = analyze_segment(
        "seg",
        front,
        model_name="yolo26s.pt",
        sample_rate_fps=5,
    )
    monkeypatch.setattr(
        "backend.app.analysis.pipeline.lane_reference_x",
        lambda *_args, **_kwargs: None,
    )
    without_lanes = analyze_segment(
        "seg",
        front,
        model_name="yolo26s.pt",
        sample_rate_fps=5,
    )

    assert len(with_lanes.events) == 1
    assert len(without_lanes.events) == 1
    assert "lane_geometry" in with_lanes.events[0].reason_labels
    assert "lane_geometry" not in without_lanes.events[0].reason_labels
    assert with_lanes.events[0].lane_change_score > without_lanes.events[0].lane_change_score
