from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from .config import get_settings

TESLA_FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})-"
    r"(?P<camera>front|back|left_repeater|right_repeater)\.mp4$",
    re.IGNORECASE,
)

CAMERA_LABELS = {
    "front": "前视",
    "back": "后视",
    "left_repeater": "左侧",
    "right_repeater": "右侧",
}


@dataclass(frozen=True)
class TeslaClipInfo:
    path: Path
    camera: str
    starts_at: datetime


@dataclass(frozen=True)
class VideoProbe:
    duration_s: float | None
    fps: float | None
    width: int | None
    height: int | None


def parse_tesla_filename(
    path: str | Path,
    *,
    offset_minutes: float | None = None,
) -> TeslaClipInfo | None:
    """Parse a Tesla Dashcam name.

    The clock in the filename is local wall time. It is not UTC. Optional
    ``offset_minutes`` (or ``BLINKLANE_CLOCK_OFFSET_MINUTES``) is added to
    that clock. The returned datetime stays naive.
    """

    file_path = Path(path)
    match = TESLA_FILENAME_RE.match(file_path.name)
    if not match:
        return None
    raw = f"{match.group('date')} {match.group('time').replace('-', ':')}"
    starts_at = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    if offset_minutes is None:
        offset_minutes = get_settings().clock_offset_minutes
    if offset_minutes:
        starts_at += timedelta(minutes=offset_minutes)
    return TeslaClipInfo(path=file_path, camera=match.group("camera").lower(), starts_at=starts_at)


def find_tesla_clips(folder_path: str | Path) -> list[TeslaClipInfo]:
    folder = Path(folder_path).expanduser().resolve()
    clips: list[TeslaClipInfo] = []
    for file_path in sorted(folder.rglob("*.mp4")):
        info = parse_tesla_filename(file_path)
        if info:
            clips.append(info)
    return clips


def probe_video(path: str | Path) -> VideoProbe:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoProbe(duration_s=None, fps=None, width=None, height=None)
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    cap.release()
    duration = frame_count / fps if fps and frame_count else None
    return VideoProbe(duration_s=duration, fps=fps, width=width, height=height)


def new_id() -> str:
    return uuid.uuid4().hex
