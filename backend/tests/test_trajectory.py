from __future__ import annotations

from backend.app.analysis.trajectory import (
    LANE_CROSS_BONUS,
    LATERAL_SHIFT_THRESHOLD,
    detect_lane_change_events,
)
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


def test_ego_motion_compensation_drops_the_same_pixel_shift() -> None:
    samples = [make_sample(idx, 180 + idx * 11) for idx in range(18)]
    series = SignalSeries(
        timestamps_s=[idx * 0.2 for idx in range(18)],
        left_energy=[0.01] * 18,
        right_energy=[0.01] * 18,
    )
    centers = {sample.timestamp_s: sample.bottom_center[0] - 40 for sample in samples}

    plain = detect_lane_change_events(samples, {7: series})
    compensated = detect_lane_change_events(
        samples,
        {7: series},
        lane_center_by_timestamp=centers,
    )

    assert LATERAL_SHIFT_THRESHOLD == 0.12
    assert len(plain) == 1
    assert compensated == []


def test_crossing_a_stable_lane_increases_the_score() -> None:
    samples = [make_sample(idx, 250 + idx * 8) for idx in range(18)]
    series = SignalSeries(
        timestamps_s=[sample.timestamp_s for sample in samples],
        left_energy=[0.01] * len(samples),
        right_energy=[0.01] * len(samples),
    )
    centers = {sample.timestamp_s: 320.0 for sample in samples}

    plain = detect_lane_change_events(samples, {7: series})
    crossed = detect_lane_change_events(
        samples,
        {7: series},
        lane_center_by_timestamp=centers,
    )

    assert len(plain) == 1
    assert len(crossed) == 1
    assert "lane_geometry" in crossed[0].reason_labels
    assert crossed[0].lane_change_score > plain[0].lane_change_score
    assert crossed[0].lane_change_score == round(
        min(1.0, plain[0].lane_change_score + LANE_CROSS_BONUS),
        3,
    )
