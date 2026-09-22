from __future__ import annotations

import json
import zipfile
from pathlib import Path

from backend.app.exporter import export_evidence_package


def test_export_evidence_package(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VCY_DATA_DIR", str(tmp_path / "data"))
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    key = tmp_path / "key.jpg"
    key.write_bytes(b"jpg")
    event = {
        "id": "evt1",
        "session_id": "sess1",
        "segment_id": "seg1",
        "camera": "front",
        "track_id": 3,
        "vehicle_type": "car",
        "start_s": 1.0,
        "end_s": 5.0,
        "key_s": 3.0,
        "lane_change_score": 0.8,
        "turn_signal_score": 0.0,
        "confidence": 0.8,
        "reason_labels": ["lane_change_candidate", "turn_signal_not_observed"],
        "review_status": "confirmed",
        "location": "测试路段",
        "note": "人工确认",
        "raw_clip_path": str(raw),
        "annotated_clip_path": None,
        "key_frame_path": str(key),
    }
    segment = {
        "path": "/videos/2026-05-25_18-30-02-front.mp4",
        "starts_at": "2026-05-25T18:30:02",
    }

    package_dir, zip_path = export_evidence_package(event, segment)

    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    report = (package_dir / "report.txt").read_text(encoding="utf-8")
    assert metadata["event_id"] == "evt1"
    assert metadata["absolute_time"] == "2026-05-25 18:30:03 至 2026-05-25 18:30:07"
    assert "绝对时间：2026-05-25 18:30:03 至 2026-05-25 18:30:07" in report
    assert "不是 UTC" in report
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert "metadata.json" in archive.namelist()
        assert "report.txt" in archive.namelist()
        assert "raw_clip.mp4" in archive.namelist()
