from __future__ import annotations

import numpy as np

from backend.app.analysis.signals import (
    detect_turn_signal,
    merge_aligned_repeater_energy,
    signal_energy_from_frame,
)
from backend.app.analysis.types import SignalSeries


def test_detect_turn_signal_observed_from_pulses() -> None:
    series = SignalSeries()
    for idx in range(30):
        series.timestamps_s.append(idx * 0.2)
        pulse = 0.8 if idx % 6 in (1, 2) else 0.02
        series.left_energy.append(pulse)
        series.right_energy.append(0.01)

    score, label = detect_turn_signal(series, 0, 6)

    assert label == "turn_signal_observed"
    assert score > 0.7


def test_detect_turn_signal_not_observed_from_flat_dark_series() -> None:
    series = SignalSeries(
        timestamps_s=[idx * 0.2 for idx in range(20)],
        left_energy=[0.01] * 20,
        right_energy=[0.02] * 20,
    )

    score, label = detect_turn_signal(series, 0, 4)

    assert label == "turn_signal_not_observed"
    assert score == 0.0


def _flat_series() -> SignalSeries:
    return SignalSeries(
        timestamps_s=[idx * 0.2 for idx in range(20)],
        left_energy=[0.01] * 20,
        right_energy=[0.01] * 20,
    )


def test_repeater_blink_counts_only_when_time_aligned() -> None:
    aligned = _flat_series()
    pulses = [(0.2, 0.9), (0.4, 0.9), (1.4, 0.9), (1.6, 0.9), (2.6, 0.9), (2.8, 0.9)]
    assert merge_aligned_repeater_energy(aligned, pulses, side="left") is True
    score, label = detect_turn_signal(aligned, 0, 4)
    assert label == "turn_signal_observed"
    assert score > 0.7

    missed = _flat_series()
    late = [(9.0, 0.9), (9.2, 0.9), (10.2, 0.9), (10.4, 0.9)]
    assert merge_aligned_repeater_energy(missed, late, side="right") is False
    score, label = detect_turn_signal(missed, 0, 4)
    assert label == "turn_signal_not_observed"
    assert score == 0.0


def test_signal_energy_from_frame_detects_amber_side_region() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[45:80, 35:45] = (0, 180, 255)

    left, right = signal_energy_from_frame(frame, (20, 20, 120, 90))

    assert left > 0.05
    assert right < left
