"""Synthetic Tesla-style clips for tests. No real dashcam footage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FRAME_SIZE = (640, 360)
FPS = 10
FRAME_COUNT = 40
CAR_TRACK_ID = 7


def write_moving_car_clip(path: Path, *, amber: bool = False) -> None:
    """Front view: a car shifts right across the lower half of the frame."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write synthetic clip: {path}")
    width, height = FRAME_SIZE
    for index in range(FRAME_COUNT):
        frame = np.full((height, width, 3), 28, dtype=np.uint8)
        x = int(150 + (index // 2) * 12)
        y2 = int(height * 0.84)
        y1 = y2 - 80
        color = (0, 180, 255) if amber else (180, 70, 30)
        cv2.rectangle(frame, (x - 35, y1), (x + 35, y2), color, thickness=-1)
        writer.write(frame)
    writer.release()


def write_blinking_repeater(path: Path) -> None:
    """Side camera with amber pulses on the same timeline as the front clip."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write synthetic clip: {path}")
    width, height = FRAME_SIZE
    for index in range(FRAME_COUNT):
        frame = np.full((height, width, 3), 12, dtype=np.uint8)
        if index % 10 in (2, 4):
            frame[:, :] = (0, 180, 255)
        writer.write(frame)
    writer.release()


def write_lane_crossing_clip(path: Path) -> None:
    """Static lane lines and a car that crosses the lane center."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write synthetic clip: {path}")
    width, height = FRAME_SIZE
    for index in range(FRAME_COUNT):
        frame = np.full((height, width, 3), 24, dtype=np.uint8)
        cv2.line(frame, (150, height - 1), (280, int(height * 0.42)), (230, 230, 230), 8)
        cv2.line(frame, (490, height - 1), (360, int(height * 0.42)), (230, 230, 230), 8)
        x = int(250 + (index // 2) * 6)
        y2 = int(height * 0.84)
        y1 = y2 - 70
        cv2.rectangle(frame, (x - 28, y1), (x + 28, y2), (180, 70, 30), thickness=-1)
        writer.write(frame)
    writer.release()


def write_dark_clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write synthetic clip: {path}")
    width, height = FRAME_SIZE
    frame = np.full((height, width, 3), 16, dtype=np.uint8)
    for _ in range(FRAME_COUNT):
        writer.write(frame)
    writer.release()


def write_tesla_folder(folder: Path) -> Path:
    """Write a four-camera folder. Repeaters stay dark so the front event remains."""

    stamp = "2026-05-25_18-30-02"
    front = folder / f"{stamp}-front.mp4"
    write_moving_car_clip(front)
    write_dark_clip(folder / f"{stamp}-back.mp4")
    write_dark_clip(folder / f"{stamp}-left_repeater.mp4")
    write_dark_clip(folder / f"{stamp}-right_repeater.mp4")
    return front


class _Values:
    def __init__(self, values: list) -> None:
        self._values = values

    def cpu(self) -> _Values:
        return self

    def int(self) -> _Values:
        return self

    def tolist(self) -> list:
        return self._values


class _Boxes:
    def __init__(self, xyxy: tuple[float, float, float, float]) -> None:
        self.xyxy = _Values([list(xyxy)])
        self.id = _Values([CAR_TRACK_ID])
        self.cls = _Values([2])

    @property
    def is_track(self) -> bool:
        return True


class _Result:
    def __init__(self, xyxy: tuple[float, float, float, float]) -> None:
        self.boxes = _Boxes(xyxy)
        self.names = {2: "car"}


class MovingCarDetector:
    """Stand-in for YOLO.track. Boxes follow a scripted path; no weights are loaded."""

    def __init__(self, x0: float = 150, step_px: float = 12) -> None:
        self.calls = 0
        self.x0 = x0
        self.step_px = step_px

    def track(self, frame: np.ndarray, **_kwargs):
        height = frame.shape[0]
        x = self.x0 + self.calls * self.step_px
        self.calls += 1
        y2 = int(height * 0.84)
        y1 = y2 - 80
        box = (float(x - 35), float(y1), float(x + 35), float(y2))
        return [_Result(box)]


def install_fake_detector(
    monkeypatch,
    *,
    x0: float = 150,
    step_px: float = 12,
) -> None:
    monkeypatch.setattr("backend.app.analysis.pipeline.yolo_available", lambda: True)
    monkeypatch.setattr(
        "backend.app.analysis.pipeline.build_model",
        lambda _model_name: MovingCarDetector(x0=x0, step_px=step_px),
    )
