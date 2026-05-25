from __future__ import annotations

import numpy as np

from backend.app.analysis.signals import detect_turn_signal, signal_energy_from_frame
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


def test_signal_energy_from_frame_detects_amber_side_region() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[45:80, 35:45] = (0, 180, 255)

    left, right = signal_energy_from_frame(frame, (20, 20, 120, 90))

    assert left > 0.05
    assert right < left
