from __future__ import annotations

import math
from statistics import median

import cv2
import numpy as np


def detect_lane_overlay(frame: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.45) :, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=max(30, w // 12),
        maxLineGap=40,
    )
    overlay = frame.copy()
    normalized: list[tuple[int, int, int, int]] = []
    if lines is None:
        return overlay, normalized
    # OpenCV 4 returns (N, 1, 4). OpenCV 5 returns (N, 4).
    rows = lines.reshape(-1, 4)
    y_offset = int(h * 0.45)
    for raw in rows[:24]:
        x1, y1, x2, y2 = (int(value) for value in raw)
        y1 += y_offset
        y2 += y_offset
        slope = (y2 - y1) / max(1, abs(x2 - x1))
        if abs(slope) < 0.35:
            continue
        normalized.append((int(x1), int(y1), int(x2), int(y2)))
        cv2.line(overlay, (x1, y1), (x2, y2), (17, 130, 108), 2)
    return overlay, normalized


def lane_reference_x(
    lines: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> float | None:
    """Lane-center x at the bottom of the frame.

    Short edges (a vehicle outline) are ignored. Both a left and a right line
    are required, and they must be far enough apart to be a lane rather than
    one object.
    """

    if width <= 0 or height <= 0 or not lines:
        return None
    bottom_y = float(height - 1)
    min_length = height * 0.28
    left_xs: list[float] = []
    right_xs: list[float] = []
    for x1, y1, x2, y2 in lines:
        if math.hypot(x2 - x1, y2 - y1) < min_length or y1 == y2:
            continue
        x_at_bottom = x1 + (bottom_y - y1) * (x2 - x1) / (y2 - y1)
        if (x1 + x2) / 2 < width / 2:
            left_xs.append(x_at_bottom)
        else:
            right_xs.append(x_at_bottom)
    if not left_xs or not right_xs:
        return None
    left = float(median(left_xs))
    right = float(median(right_xs))
    if right - left < width * 0.18:
        return None
    return (left + right) / 2.0
