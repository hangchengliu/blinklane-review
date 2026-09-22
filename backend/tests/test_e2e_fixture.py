"""Import → analyze → confirm → evidence zip, without real Tesla footage or YOLO weights."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.tests.synthetic import install_fake_detector, write_tesla_folder


def test_fixture_import_analyze_confirm_and_export(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    folder = tmp_path / "TeslaCam" / "SavedClips" / "2026-05-25_18-30-02"
    write_tesla_folder(folder)
    install_fake_detector(monkeypatch)
    client = TestClient(app)

    imported = client.post("/api/import", json={"folder_path": str(folder)})
    assert imported.status_code == 200
    session_id = imported.json()["session"]["id"]
    assert {item["camera"] for item in imported.json()["segments"]} == {
        "front",
        "back",
        "left_repeater",
        "right_repeater",
    }

    created = client.post("/api/analyze", json={"session_id": session_id})
    job = client.get(f"/api/jobs/{created.json()['id']}")
    events = client.get("/api/events", params={"session_id": session_id})

    assert job.status_code == 200
    assert job.json()["status"] == "completed", job.json()["message"]
    body = events.json()
    assert len(body) == 1
    event = body[0]
    assert event["track_id"] == 7
    assert event["vehicle_type"] == "car"
    assert "turn_signal_not_observed" in event["reason_labels"]
    assert event["lane_change_score"] >= 0.9
    assert event["confidence"] > 0.5

    review = client.patch(
        f"/api/events/{event['id']}/review",
        json={"review_status": "confirmed", "location": "合成路段", "note": "夹具确认"},
    )
    assert review.status_code == 200
    assert review.json()["review_status"] == "confirmed"

    exported = client.post(f"/api/events/{event['id']}/export")
    assert exported.status_code == 200
    download = client.get(exported.json()["download_url"])
    assert download.status_code == 200

    archive_path = tmp_path / "evidence.zip"
    archive_path.write_bytes(download.content)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "metadata.json" in names
        assert "report.txt" in names
        metadata = json.loads(archive.read("metadata.json"))
        report = archive.read("report.txt").decode("utf-8")

    assert metadata["review_status"] == "confirmed"
    assert metadata["location"] == "合成路段"
    assert "2026-05-25 18:30:02 至 2026-05-25 18:30:07" in metadata["absolute_time"]
    assert "2026-05-25 18:30:02 至 2026-05-25 18:30:07" in report
    assert "合成路段" in report
