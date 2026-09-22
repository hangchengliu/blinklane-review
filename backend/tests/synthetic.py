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
    """Stand-in for YOLO.track. Boxes follow the synthetic car; no weights are loaded."""

    def __init__(self) -> None:
        self.calls = 0

    def track(self, frame: np.ndarray, **_kwargs):
        height = frame.shape[0]
        step = self.calls
        self.calls += 1
        x = 150 + step * 12
        y2 = int(height * 0.84)
        y1 = y2 - 80
        box = (float(x - 35), float(y1), float(x + 35), float(y2))
        return [_Result(box)]


def install_fake_detector(monkeypatch) -> MovingCarDetector:
    detector = MovingCarDetector()
    monkeypatch.setattr("backend.app.analysis.pipeline.yolo_available", lambda: True)
    monkeypatch.setattr(
        "backend.app.analysis.pipeline.build_model",
        lambda _model_name: detector,
    )
    return detector
