from __future__ import annotations

import importlib.util
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..config import Settings, get_settings
from ..tesla import new_id
from ..video import create_annotated_clip, extract_raw_clip, grab_key_frame
from .lane import detect_lane_lines, lane_reference_x
from .signals import merge_aligned_repeater_energy, repeater_amber_energy, signal_energy_from_frame
from .trajectory import detect_lane_change_events
from .types import EventCandidate, SignalSeries, TrackSample

VEHICLE_CLASS_IDS = [2, 3, 5, 7]

logger = logging.getLogger(__name__)


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
    repeater_paths: dict[str, Path] | None = None,
) -> SegmentAnalysis:
    if not yolo_available():
        raise RuntimeError(
            "Ultralytics YOLO is not installed. Run: python -m pip install -e '.[yolo]'"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        logger.warning("Video %s reports no FPS; assuming 30", video_path)
        fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(round(fps / sample_rate_fps)))
    model = build_model(model_name)
    device = choose_torch_device()

    samples: list[TrackSample] = []
    signal_series_by_track: dict[int, SignalSeries] = defaultdict(SignalSeries)
    lane_center_by_timestamp: dict[float, float] = {}
    frame_index = 0
    processed = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        # Known trade-off: the tracker runs with persist=True but only sees every
        # frame_step-th frame, so very fast lateral motion between samples can
        # switch or drop track IDs. Dashcam motion at the default sample rate is
        # slow enough that this is acceptable, and it keeps analysis cheap.
        result = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASS_IDS,
            conf=0.25,
            iou=0.5,
            device=device,
            verbose=False,
        )[0]
        timestamp_s = frame_index / fps
        lane_lines = detect_lane_lines(frame)
        lane_center = lane_reference_x(lane_lines, frame.shape[1], frame.shape[0])
        if lane_center is not None:
            lane_center_by_timestamp[timestamp_s] = lane_center
        if result.boxes is not None and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            names = result.names
            for bbox, track_id, class_id in zip(boxes, track_ids, class_ids, strict=False):
                # Detector glitches can yield NaN/inf boxes; skip those samples
                # here so a single bad box cannot kill the whole analysis job.
                if not all(math.isfinite(float(value)) for value in bbox):
                    continue
                class_name = str(names.get(class_id, class_id))
                sample = TrackSample(
                    track_id=int(track_id),
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
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

    warnings: list[str] = []
    warnings.extend(_merge_repeater_signals(signal_series_by_track, repeater_paths or {}))
    events = detect_lane_change_events(
        samples,
        signal_series_by_track,
        lane_center_by_timestamp=lane_center_by_timestamp,
    )
    settings = get_settings()
    for event in events:
        warnings.extend(attach_event_assets(segment_id, video_path, event, settings))
    return SegmentAnalysis(events=events, warnings=warnings)


def _merge_repeater_signals(
    signal_series_by_track: dict[int, SignalSeries],
    repeater_paths: dict[str, Path],
) -> list[str]:
    """Blend left/right repeater amber energy into each track at the same timestamps.

    Repeater energy is evidence: it may raise a track's turn-signal score, but
    trajectory.py caps the label at "inconclusive" unless the front camera also
    saw amber, so a blinking side vehicle cannot by itself clear an event.
    """

    warnings: list[str] = []
    if not signal_series_by_track or not repeater_paths:
        return warnings
    # Sample on the union of ALL tracks' timestamps: the old first-track grid
    # silently disabled repeater evidence for every other track when the first
    # track was short.
    timestamps = _union_timestamps(signal_series_by_track)
    merged: dict[str, list[tuple[float, float]]] = {}
    for camera, side in (("left_repeater", "left"), ("right_repeater", "right")):
        path = repeater_paths.get(camera)
        if path is None:
            continue
        try:
            merged[side] = _read_repeater_energies(path, timestamps)
        except Exception as exc:
            warnings.append(f"{camera} failed: {exc}")
    for series in signal_series_by_track.values():
        # Snapshot front-camera energy before the repeater merge so an
        # "observed" label can demand front-camera corroboration later.
        series.front_energy = [
            max(left, right)
            for left, right in zip(series.left_energy, series.right_energy, strict=False)
        ]
        for side, samples in merged.items():
            merge_aligned_repeater_energy(series, samples, side=side)
    return warnings


def _union_timestamps(signal_series_by_track: dict[int, SignalSeries]) -> list[float]:
    """Sorted union of every track's timestamps.

    Timestamps closer than half the smallest sampling period are treated as the
    same instant, so tracks on the same clock do not produce near-duplicates.
    """

    all_timestamps = sorted(
        timestamp for series in signal_series_by_track.values() for timestamp in series.timestamps_s
    )
    if not all_timestamps:
        return []
    periods = [
        later - earlier
        for series in signal_series_by_track.values()
        for earlier, later in zip(series.timestamps_s, series.timestamps_s[1:], strict=False)
        if later > earlier
    ]
    epsilon = min(periods) / 2 if periods else 0.05
    union = [all_timestamps[0]]
    for timestamp in all_timestamps[1:]:
        if timestamp - union[-1] > epsilon:
            union.append(timestamp)
    return union


def _read_repeater_energies(path: Path, timestamps: list[float]) -> list[tuple[float, float]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open repeater video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    samples: list[tuple[float, float]] = []
    try:
        # Frame-index convention is floor(int(timestamp * fps)), matching video.py.
        by_index: dict[int, list[float]] = defaultdict(list)
        for timestamp in timestamps:
            by_index[max(0, int(timestamp * fps))].append(timestamp)
        indices = sorted(by_index)
        if indices:
            # Timestamps are sorted, so seek once to the first needed frame and
            # walk forward with grab()/retrieve() pairs; random-access seeks per
            # timestamp are slow and unreliable with some codecs.
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
            next_index = indices[0]
            for frame_index in indices:
                while next_index < frame_index and cap.grab():
                    next_index += 1
                frame = None
                if next_index == frame_index and cap.grab():
                    ok, frame = cap.retrieve()
                    if not ok:
                        frame = None
                    next_index += 1
                energy = repeater_amber_energy(frame) if frame is not None else 0.0
                samples.extend((timestamp, energy) for timestamp in by_index[frame_index])
    finally:
        cap.release()
    return samples


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
