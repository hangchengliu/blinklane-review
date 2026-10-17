"""Unit tests for lane line detection and the lane-center reference."""

from __future__ import annotations

import cv2
import numpy as np

from backend.app.analysis.lane import (
    detect_lane_lines,
    detect_lane_overlay,
    lane_reference_x,
)

WIDTH = 640
HEIGHT = 400


def _two_lane_frame() -> np.ndarray:
    frame = np.full((360, 640, 3), 24, dtype=np.uint8)
    cv2.line(frame, (150, 359), (280, 151), (230, 230, 230), 8)
    cv2.line(frame, (490, 359), (360, 151), (230, 230, 230), 8)
    return frame


def test_detect_lane_lines_blank_frame_returns_nothing() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    assert detect_lane_lines(frame) == []


def test_detect_lane_lines_finds_both_sides() -> None:
    lines = detect_lane_lines(_two_lane_frame())

    assert lines
    mids = [(x1 + x2) / 2 for x1, _y1, x2, _y2 in lines]
    assert any(mid < WIDTH / 2 for mid in mids)
    assert any(mid >= WIDTH / 2 for mid in mids)
    center = lane_reference_x(lines, WIDTH, 360)
    assert center is not None
    assert 250 < center < 390


def test_detect_lane_overlay_is_a_thin_drawing_wrapper() -> None:
    frame = _two_lane_frame()

    overlay, lines = detect_lane_overlay(frame)

    assert lines == detect_lane_lines(frame)
    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, frame)


def test_lane_reference_x_no_lines() -> None:
    assert lane_reference_x([], WIDTH, HEIGHT) is None


def test_lane_reference_x_one_sided_lines() -> None:
    lines = [(150, 399, 280, 168)]

    assert lane_reference_x(lines, WIDTH, HEIGHT) is None


def test_lane_reference_x_lines_shorter_than_min_length_gate() -> None:
    # min length is height * 0.28 = 112; the left line is only ~56 px long.
    lines = [(150, 399, 175, 349), (490, 399, 360, 168)]

    assert lane_reference_x(lines, WIDTH, HEIGHT) is None


def test_lane_reference_x_lines_closer_than_min_separation() -> None:
    # min separation is width * 0.18 = 115.2; these bottoms are 50 px apart.
    lines = [(300, 399, 330, 168), (350, 399, 380, 168)]

    assert lane_reference_x(lines, WIDTH, HEIGHT) is None


def test_lane_reference_x_vertical_lines_are_valid() -> None:
    lines = [(100, 50, 100, 380), (540, 50, 540, 380)]

    assert lane_reference_x(lines, WIDTH, HEIGHT) == 320.0


def test_lane_reference_x_normal_two_lane_case() -> None:
    lines = [(150, 399, 280, 168), (490, 399, 360, 168)]

    assert lane_reference_x(lines, WIDTH, HEIGHT) == 320.0
