from __future__ import annotations

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
    for raw in lines[:24]:
        x1, y1, x2, y2 = raw[0]
        y1 += int(h * 0.45)
        y2 += int(h * 0.45)
        slope = (y2 - y1) / max(1, abs(x2 - x1))
        if abs(slope) < 0.35:
            continue
        normalized.append((int(x1), int(y1), int(x2), int(y2)))
        cv2.line(overlay, (x1, y1), (x2, y2), (17, 130, 108), 2)
    return overlay, normalized
