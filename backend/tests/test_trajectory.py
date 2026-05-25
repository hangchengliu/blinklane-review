from __future__ import annotations

from backend.app.analysis.trajectory import detect_lane_change_events
from backend.app.analysis.types import SignalSeries, TrackSample


def make_sample(idx: int, x: float) -> TrackSample:
    return TrackSample(
        track_id=7,
        frame_index=idx,
        timestamp_s=idx * 0.2,
        bbox_xyxy=(x - 30, 280, x + 30, 360),
        class_name="car",
        frame_width=640,
        frame_height=400,
    )


def test_detect_lane_change_without_signal() -> None:
    samples = [make_sample(idx, 180 + idx * 11) for idx in range(18)]
    series = SignalSeries(
        timestamps_s=[idx * 0.2 for idx in range(18)],
        left_energy=[0.01] * 18,
        right_energy=[0.01] * 18,
    )

    events = detect_lane_change_events(samples, {7: series})

    assert len(events) == 1
    assert events[0].track_id == 7
    assert events[0].turn_signal_score == 0.0
    assert "turn_signal_not_observed" in events[0].reason_labels


def test_lane_change_with_observed_signal_is_suppressed() -> None:
    samples = [make_sample(idx, 180 + idx * 11) for idx in range(18)]
    series = SignalSeries()
    for idx in range(18):
        series.timestamps_s.append(idx * 0.2)
        series.left_energy.append(0.75 if idx % 5 in (1, 2) else 0.02)
        series.right_energy.append(0.01)

    events = detect_lane_change_events(samples, {7: series})

    assert events == []
