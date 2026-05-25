from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .analysis.lane import detect_lane_overlay
from .analysis.types import EventCandidate, TrackSample


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_raw_clip(source: Path, start_s: float, end_s: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.2, end_s - start_s)
    if ffmpeg_available():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{max(0.0, start_s):.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-c",
                "copy",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    _extract_clip_with_opencv(source, start_s, end_s, out_path)


def grab_key_frame(
    source: Path,
    timestamp_s: float,
    out_path: Path,
    sample: TrackSample | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(timestamp_s * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Frame unavailable",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    if sample:
        draw_sample(frame, sample, "key frame")
    cv2.imwrite(str(out_path), frame)


def create_annotated_clip(source: Path, event: EventCandidate, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    start_frame = max(0, int(event.start_s * fps))
    end_frame = max(start_frame + 1, int(event.end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    samples_by_frame = {sample.frame_index: sample for sample in event.samples}
    frame_index = start_frame
    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        overlay, _ = detect_lane_overlay(frame)
        frame = cv2.addWeighted(frame, 0.78, overlay, 0.22, 0)
        closest = _nearest_sample(samples_by_frame, frame_index)
        if closest:
            draw_sample(frame, closest, f"suspected lane change #{event.track_id}")
        label = (
            f"lane={event.lane_change_score:.2f} "
            f"signal={event.turn_signal_score:.2f} "
            f"confidence={event.confidence:.2f}"
        )
        cv2.putText(frame, label, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()


def draw_sample(frame: np.ndarray, sample: TrackSample, label: str) -> None:
    x1, y1, x2, y2 = (int(value) for value in sample.bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 96, 255), 3)
    cv2.putText(
        frame,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 96, 255),
        2,
    )


def _extract_clip_with_opencv(source: Path, start_s: float, end_s: float, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_s * fps)))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    current_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    while current_frame <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        current_frame += 1
    writer.release()
    cap.release()


def _nearest_sample(
    samples_by_frame: dict[int, TrackSample],
    frame_index: int,
) -> TrackSample | None:
    if frame_index in samples_by_frame:
        return samples_by_frame[frame_index]
    if not samples_by_frame:
        return None
    nearest_frame = min(samples_by_frame, key=lambda idx: abs(idx - frame_index))
    if abs(nearest_frame - frame_index) > 10:
        return None
    return samples_by_frame[nearest_frame]
