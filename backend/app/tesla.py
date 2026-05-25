from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2

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


def parse_tesla_filename(path: str | Path) -> TeslaClipInfo | None:
    file_path = Path(path)
    match = TESLA_FILENAME_RE.match(file_path.name)
    if not match:
        return None
    raw = f"{match.group('date')} {match.group('time').replace('-', ':')}"
    starts_at = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
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
