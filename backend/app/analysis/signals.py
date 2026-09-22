from __future__ import annotations

import math

import cv2
import numpy as np

from .types import BBox, SignalSeries


def _clip_bbox(frame: np.ndarray, bbox: BBox) -> tuple[int, int, int, int] | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def signal_energy_from_frame(frame: np.ndarray, bbox: BBox) -> tuple[float, float]:
    clipped = _clip_bbox(frame, bbox)
    if clipped is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0.0

    # Tesla front camera usually sees the rear of the target car. Use the lower side bands
    # where rear turn indicators are most likely to appear.
    y_start = int(h * 0.35)
    y_end = int(h * 0.9)
    left_roi = crop[y_start:y_end, : max(1, int(w * 0.38))]
    right_roi = crop[y_start:y_end, min(w - 1, int(w * 0.62)) :]
    return _amber_energy(left_roi), _amber_energy(right_roi)


def _amber_energy(roi: np.ndarray) -> float:
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([8, 70, 80], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    bright = hsv[:, :, 2] > 120
    score = float(np.mean((mask > 0) & bright))
    return max(0.0, min(1.0, score * 8.0))


def merge_aligned_repeater_energy(
    series: SignalSeries,
    repeater_samples: list[tuple[float, float]],
    *,
    side: str,
    tolerance_s: float = 0.15,
) -> bool:
    """Max-merge repeater amber energy into one side of the front-camera series.

    ``repeater_samples`` are ``(timestamp_s, energy)`` on the repeater clip's own
    clock, which shares the front clip's start. A sample is used only when its
    timestamp is within ``tolerance_s`` of a front sample, so a blink from
    another moment does not count.
    """

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    target = series.left_energy if side == "left" else series.right_energy
    used = False
    for index, timestamp in enumerate(series.timestamps_s):
        energy = _nearest_repeater_energy(repeater_samples, timestamp, tolerance_s)
        if energy > target[index]:
            target[index] = energy
            used = True
    return used


def _nearest_repeater_energy(
    samples: list[tuple[float, float]],
    timestamp_s: float,
    tolerance_s: float,
) -> float:
    if not samples:
        return 0.0
    nearest_time, energy = min(samples, key=lambda item: abs(item[0] - timestamp_s))
    if abs(nearest_time - timestamp_s) > tolerance_s:
        return 0.0
    return energy


def repeater_amber_energy(frame: np.ndarray) -> float:
    """Amber energy across the repeater view, which looks out to the side."""

    if frame.size == 0:
        return 0.0
    height, width = frame.shape[:2]
    if height < 8 or width < 8:
        return 0.0
    roi = frame[int(height * 0.2) : int(height * 0.9), int(width * 0.1) : int(width * 0.9)]
    return _amber_energy(roi)


def detect_turn_signal(series: SignalSeries, start_s: float, end_s: float) -> tuple[float, str]:
    values: list[float] = []
    for timestamp, left, right in zip(
        series.timestamps_s, series.left_energy, series.right_energy, strict=False
    ):
        if start_s <= timestamp <= end_s:
            values.append(max(left, right))

    if len(values) < 8:
        return 0.5, "turn_signal_inconclusive"

    arr = np.asarray(values, dtype=np.float32)
    dynamic_range = float(np.max(arr) - np.min(arr))
    peak_count = _count_peaks(arr)
    high_ratio = float(np.mean(arr > max(0.12, float(np.mean(arr) + np.std(arr) * 0.5))))

    if dynamic_range > 0.18 and peak_count >= 2 and 0.08 <= high_ratio <= 0.65:
        score = min(1.0, 0.45 + dynamic_range + peak_count * 0.08)
        return score, "turn_signal_observed"

    if float(np.max(arr)) < 0.12 or dynamic_range < 0.08:
        return 0.0, "turn_signal_not_observed"

    return 0.35, "turn_signal_inconclusive"


def _count_peaks(arr: np.ndarray) -> int:
    if arr.size < 3:
        return 0
    threshold = max(0.1, float(np.mean(arr) + np.std(arr) * 0.4))
    peaks = 0
    last_peak = -math.inf
    for idx in range(1, arr.size - 1):
        if arr[idx] >= threshold and arr[idx] > arr[idx - 1] and arr[idx] >= arr[idx + 1]:
            if idx - last_peak >= 2:
                peaks += 1
                last_peak = idx
    return peaks
