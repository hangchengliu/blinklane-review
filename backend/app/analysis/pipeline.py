from __future__ import annotations

import importlib.util
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..config import Settings, get_settings
from ..tesla import new_id
from ..video import create_annotated_clip, extract_raw_clip, grab_key_frame
from .signals import signal_energy_from_frame
from .trajectory import detect_lane_change_events
from .types import EventCandidate, SignalSeries, TrackSample

VEHICLE_CLASS_IDS = [2, 3, 5, 7]


@dataclass
class SegmentAnalysis:
    events: list[EventCandidate]
    warnings: list[str]


def yolo_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def build_model(model_name: str):
    from ultralytics import YOLO

    return YOLO(model_name)


def choose_torch_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""

    try:
        import torch
    except Exception:
        return "cpu"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def analyze_segment(
    segment_id: str,
    video_path: Path,
    *,
    model_name: str,
    sample_rate_fps: float,
    progress: Callable[[float, str], None] | None = None,
) -> SegmentAnalysis:
    if not yolo_available():
        raise RuntimeError(
            "Ultralytics YOLO is not installed. Run: python -m pip install -e '.[yolo]'"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(round(fps / sample_rate_fps)))
    model = build_model(model_name)
    device = choose_torch_device()

    samples: list[TrackSample] = []
    signal_series_by_track: dict[int, SignalSeries] = defaultdict(SignalSeries)
    frame_index = 0
    processed = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        result = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASS_IDS,
            conf=0.25,
            iou=0.5,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is not None and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            names = result.names
            for bbox, track_id, class_id in zip(boxes, track_ids, class_ids, strict=False):
                class_name = str(names.get(class_id, class_id))
                sample = TrackSample(
                    track_id=int(track_id),
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                    bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
                    class_name=class_name,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )
                samples.append(sample)
                left, right = signal_energy_from_frame(frame, sample.bbox_xyxy)
                series = signal_series_by_track[sample.track_id]
                series.timestamps_s.append(sample.timestamp_s)
                series.left_energy.append(left)
                series.right_energy.append(right)

        processed += 1
        if progress and processed % 10 == 0:
            denominator = max(1, frame_count)
            progress(min(0.95, frame_index / denominator), f"Analyzed frame {frame_index}")
        frame_index += 1
    cap.release()

    events = detect_lane_change_events(samples, signal_series_by_track)
    settings = get_settings()
    warnings: list[str] = []
    for event in events:
        warnings.extend(attach_event_assets(segment_id, video_path, event, settings))
    return SegmentAnalysis(events=events, warnings=warnings)


def attach_event_assets(
    segment_id: str,
    video_path: Path,
    event: EventCandidate,
    settings: Settings | None = None,
) -> list[str]:
    """Write clip and key-frame files. Failures become warnings, not empty paths."""

    settings = settings or get_settings()
    warnings: list[str] = []
    asset_stem = f"{segment_id}_{event.track_id}_{new_id()[:8]}"
    raw_path = settings.data_dir / "clips" / f"{asset_stem}_raw.mp4"
    annotated_path = settings.data_dir / "annotated" / f"{asset_stem}_annotated.mp4"
    key_path = settings.data_dir / "keys" / f"{asset_stem}_key.jpg"
    target_sample = _nearest_sample_by_time(event.samples, event.key_s)

    def write_raw(path: Path) -> None:
        extract_raw_clip(video_path, event.start_s, event.end_s, path)

    def write_annotated(path: Path) -> None:
        create_annotated_clip(video_path, event, path)

    def write_key(path: Path) -> None:
        grab_key_frame(video_path, event.key_s, path, target_sample)

    writers = (
        ("raw_clip", raw_path, write_raw),
        ("annotated_clip", annotated_path, write_annotated),
        ("key_frame", key_path, write_key),
    )
    for label, path, writer in writers:
        try:
            writer(path)
        except Exception as exc:
            warnings.append(f"{label} failed for track {event.track_id}: {exc}")
            continue
        event.reason_labels.append(f"{label}={path}")
    return warnings


def _nearest_sample_by_time(samples: list[TrackSample], timestamp_s: float) -> TrackSample | None:
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(sample.timestamp_s - timestamp_s))
