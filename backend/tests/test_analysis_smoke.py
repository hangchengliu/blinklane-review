"""Analysis smoke that never downloads YOLO weights.

CI runs this file on its own. The detector is a fake; ``build_model`` must not
call Ultralytics.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app


class _EmptyDetector:
    def track(self, _frame, **_kwargs):
        return [SimpleNamespace(boxes=None, names={})]


def _write_clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(8):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (index, 32, 32)
        writer.write(frame)
    writer.release()


def test_analyze_smoke_mocks_detector_and_skips_weights(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    clip = tmp_path / "clips" / "2026-05-25_18-30-02-front.mp4"
    _write_clip(clip)
    loaded: list[str] = []

    def build_model(model_name: str) -> _EmptyDetector:
        loaded.append(model_name)
        return _EmptyDetector()

    monkeypatch.setattr("backend.app.analysis.pipeline.yolo_available", lambda: True)
    monkeypatch.setattr("backend.app.analysis.pipeline.build_model", build_model)
    client = TestClient(app)
    imported = client.post("/api/import", json={"folder_path": str(tmp_path / "clips")})
    assert imported.status_code == 200
    created = client.post(
        "/api/analyze",
        json={"session_id": imported.json()["session"]["id"]},
    )
    job = client.get(f"/api/jobs/{created.json()['id']}")

    assert created.status_code == 200
    assert job.json()["status"] == "completed"
    assert job.json()["message"] == "Analysis completed"
    assert loaded == ["yolo26s.pt"]
    assert job.json()["event_count"] == 0
