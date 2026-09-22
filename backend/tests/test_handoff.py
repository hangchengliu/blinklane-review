"""Report URL handoff after confirm (no city-site client)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.tests.synthetic import install_fake_detector, write_tesla_folder


def test_report_url_empty_by_default_on_confirm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BLINKLANE_REPORT_URL", raising=False)
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    folder = tmp_path / "TeslaCam" / "SavedClips" / "2026-05-25_18-30-02"
    write_tesla_folder(folder)
    install_fake_detector(monkeypatch)
    client = TestClient(app)

    imported = client.post("/api/import", json={"folder_path": str(folder)})
    session_id = imported.json()["session"]["id"]
    client.post("/api/analyze", json={"session_id": session_id})
    event = client.get("/api/events", params={"session_id": session_id}).json()[0]

    review = client.patch(
        f"/api/events/{event['id']}/review",
        json={"review_status": "confirmed", "location": "路段", "note": "", "plate": "X1"},
    )
    assert review.status_code == 200
    body = review.json()
    assert body.get("report_url") in (None, "")
    assert body["zip_path"]

    pending = client.patch(
        f"/api/events/{event['id']}/review",
        json={"review_status": "pending", "location": "路段", "note": "", "plate": "X1"},
    )
    assert pending.status_code == 200
    assert pending.json().get("report_url") in (None, "")
    assert "zip_path" not in pending.json()


def test_report_url_only_when_confirmed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_REPORT_URL", "https://example.test/report")
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    folder = tmp_path / "TeslaCam" / "SavedClips" / "2026-05-25_18-30-02"
    write_tesla_folder(folder)
    install_fake_detector(monkeypatch)
    client = TestClient(app)

    imported = client.post("/api/import", json={"folder_path": str(folder)})
    session_id = imported.json()["session"]["id"]
    client.post("/api/analyze", json={"session_id": session_id})
    event = client.get("/api/events", params={"session_id": session_id}).json()[0]

    pending = client.patch(
        f"/api/events/{event['id']}/review",
        json={"review_status": "pending", "location": "路段", "note": "", "plate": ""},
    )
    assert pending.json().get("report_url") in (None, "")

    confirmed = client.patch(
        f"/api/events/{event['id']}/review",
        json={"review_status": "confirmed", "location": "路段", "note": "", "plate": "Y2"},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["report_url"] == "https://example.test/report"
    assert confirmed.json()["zip_path"]
