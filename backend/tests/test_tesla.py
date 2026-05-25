from __future__ import annotations

from pathlib import Path

from backend.app.tesla import find_tesla_clips, parse_tesla_filename


def test_parse_tesla_filename() -> None:
    info = parse_tesla_filename("2026-05-25_18-30-02-front.mp4")

    assert info is not None
    assert info.camera == "front"
    assert info.starts_at.isoformat() == "2026-05-25T18:30:02+00:00"


def test_find_tesla_clips_ignores_non_matching_files(tmp_path: Path) -> None:
    (tmp_path / "2026-05-25_18-30-02-front.mp4").write_bytes(b"not real video")
    (tmp_path / "2026-05-25_18-30-02-left_repeater.mp4").write_bytes(b"not real video")
    (tmp_path / "random.mp4").write_bytes(b"not real video")

    clips = find_tesla_clips(tmp_path)

    assert [clip.camera for clip in clips] == ["front", "left_repeater"]
