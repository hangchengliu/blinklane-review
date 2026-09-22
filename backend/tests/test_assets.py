from __future__ import annotations

from pathlib import Path

from backend.app.analysis.pipeline import attach_event_assets
from backend.app.analysis.types import EventCandidate


def test_asset_failures_are_warnings_without_empty_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))

    def explode(*_args, **_kwargs) -> None:
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr("backend.app.analysis.pipeline.extract_raw_clip", explode)
    monkeypatch.setattr("backend.app.analysis.pipeline.create_annotated_clip", explode)
    monkeypatch.setattr("backend.app.analysis.pipeline.grab_key_frame", explode)
    event = EventCandidate(
        track_id=4,
        vehicle_type="car",
        start_s=0.0,
        end_s=1.0,
        key_s=0.4,
        lane_change_score=0.7,
        turn_signal_score=0.0,
        confidence=0.7,
        reason_labels=["lane_change_candidate"],
    )

    warnings = attach_event_assets("segment", Path("missing.mp4"), event)

    assert len(warnings) == 3
    assert all("encoder unavailable" in warning for warning in warnings)
    assert not any("=" in label for label in event.reason_labels)
