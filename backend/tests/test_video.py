"""Clip extraction through both backends (ffmpeg and the OpenCV fallback)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend.app.video import extract_raw_clip
from backend.tests.synthetic import write_moving_car_clip


@pytest.mark.parametrize("has_ffmpeg", [True, False])
def test_extract_raw_clip_produces_non_empty_file(
    monkeypatch,
    tmp_path: Path,
    has_ffmpeg: bool,
) -> None:
    monkeypatch.setattr("backend.app.video.ffmpeg_available", lambda: has_ffmpeg)
    source = tmp_path / "source.mp4"
    write_moving_car_clip(source)
    out = tmp_path / "out.mp4"

    extract_raw_clip(source, 0.5, 2.5, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_ffmpeg_timeout_falls_back_to_opencv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.app.video.ffmpeg_available", lambda: True)
    source = tmp_path / "source.mp4"
    write_moving_car_clip(source)

    def hang(*_args, **kwargs) -> None:
        assert kwargs.get("timeout") is not None
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)

    monkeypatch.setattr("backend.app.video.subprocess.run", hang)
    out = tmp_path / "out.mp4"

    extract_raw_clip(source, 0.5, 2.5, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_unopened_video_writer_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.app.video.ffmpeg_available", lambda: False)
    source = tmp_path / "source.mp4"
    write_moving_car_clip(source)

    class _BrokenWriter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr("backend.app.video.cv2.VideoWriter", _BrokenWriter)

    with pytest.raises(RuntimeError, match="VideoWriter"):
        extract_raw_clip(source, 0.5, 2.5, tmp_path / "out.mp4")
