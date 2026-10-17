"""Regression tests for the 2026-10-17 code-review findings in analysis/."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from backend.app.analysis.pipeline import _merge_repeater_signals, analyze_segment
from backend.app.analysis.signals import detect_turn_signal
from backend.app.analysis.trajectory import detect_lane_change_events
from backend.app.analysis.types import SignalSeries, TrackSample
from backend.tests.synthetic import (
    MovingCarDetector,
    _Result,
    write_blinking_repeater,
    write_moving_car_clip,
)


def make_sample(idx: int, x: float, track_id: int = 7) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=idx,
        timestamp_s=idx * 0.2,
        bbox_xyxy=(x - 30, 280, x + 30, 360),
        class_name="car",
        frame_width=640,
        frame_height=400,
    )


def flat_series(length: int, value: float) -> SignalSeries:
    return SignalSeries(
        timestamps_s=[idx * 0.2 for idx in range(length)],
        left_energy=[value] * length,
        right_energy=[value] * length,
    )


# Finding 1: mixed coordinate systems invented phantom shifts when lane
# detection flickered; a track must use a single reference frame.


def test_intermittent_lane_centers_do_not_create_phantom_shift() -> None:
    samples = [make_sample(idx, 300) for idx in range(18)]
    centers = {
        sample.timestamp_s: sample.bottom_center[0] - 30
        for sample in samples
        if sample.frame_index % 2 == 0
    }

    events = detect_lane_change_events(samples, {}, lane_center_by_timestamp=centers)

    assert events == []


def test_few_lane_centers_fall_back_to_absolute_positions() -> None:
    samples = [make_sample(idx, 180 + idx * 11) for idx in range(18)]
    centers = {sample.timestamp_s: 320.0 for sample in samples[:3]}

    events = detect_lane_change_events(samples, {}, lane_center_by_timestamp=centers)

    assert len(events) == 1
    assert "ego_motion_compensated" not in events[0].reason_labels


# Finding 2: a flat-BRIGHT amber series is inconclusive, not "not observed".


def test_turn_signal_pulses_are_observed() -> None:
    series = flat_series(30, 0.02)
    for idx in range(30):
        series.left_energy[idx] = 0.8 if idx % 6 in (1, 2) else 0.02

    score, label = detect_turn_signal(series, 0, 6)

    assert label == "turn_signal_observed"
    assert score > 0.7


def test_turn_signal_flat_dark_is_not_observed() -> None:
    score, label = detect_turn_signal(flat_series(20, 0.01), 0, 4)

    assert label == "turn_signal_not_observed"
    assert score == 0.0


def test_turn_signal_flat_bright_is_inconclusive() -> None:
    score, label = detect_turn_signal(flat_series(20, 0.9), 0, 4)

    assert label == "turn_signal_inconclusive"
    assert score == 0.5


# Finding 3: the repeater sampling grid is the union of all tracks' timestamps.


def test_repeater_grid_covers_timestamps_beyond_a_short_first_track(tmp_path: Path) -> None:
    repeater = tmp_path / "2026-05-25_18-30-02-left_repeater.mp4"
    write_blinking_repeater(repeater)
    short = SignalSeries(timestamps_s=[0.0], left_energy=[0.0], right_energy=[0.0])
    long = flat_series(20, 0.0)
    by_track = {1: short, 2: long}  # short track first: the old grid used only t=0.0

    warnings = _merge_repeater_signals(by_track, {"left_repeater": repeater})

    assert warnings == []
    late_hits = [
        timestamp
        for timestamp, energy in zip(long.timestamps_s, long.left_energy, strict=False)
        if timestamp >= 1.0 and energy > 0.5
    ]
    assert late_hits


# Finding 4: the signal scoring window must cover the whole event window.


def test_blink_near_track_start_still_counts_as_observed() -> None:
    samples = [make_sample(idx, 150 + idx * 8) for idx in range(40)]
    series = flat_series(40, 0.01)
    for idx in (1, 2, 6, 7):  # blink pulses only near the start (t <= 1.4)
        series.left_energy[idx] = 0.8
    # key_s is t=4.0; the old key_s +/- 2.5 window started at t=1.5 and missed
    # every pulse.

    events = detect_lane_change_events(samples, {7: series})

    assert events == []  # signal observed -> event suppressed


# Finding 5: repeater energy alone must not flip the label to "observed".


def test_repeater_only_blink_does_not_suppress_the_event(tmp_path: Path) -> None:
    repeater = tmp_path / "2026-05-25_18-30-02-left_repeater.mp4"
    write_blinking_repeater(repeater)
    samples = [make_sample(idx, 180 + idx * 11) for idx in range(18)]
    series = flat_series(18, 0.01)  # front camera stays dark
    by_track = {7: series}
    _merge_repeater_signals(by_track, {"left_repeater": repeater})

    # The merged series alone looks like a blink; the evidence is real.
    _score, merged_label = detect_turn_signal(series, 0, 4)
    assert merged_label == "turn_signal_observed"

    events = detect_lane_change_events(samples, by_track)

    assert len(events) == 1
    assert "turn_signal_observed" not in events[0].reason_labels
    assert "turn_signal_inconclusive" in events[0].reason_labels


# Finding 6: a NaN/inf bbox from the detector must not kill the analysis job.


class _NaNOnceDetector(MovingCarDetector):
    """First call returns a NaN box; later calls behave normally."""

    def track(self, frame: np.ndarray, **kwargs):
        if self.calls == 0:
            self.calls += 1
            return [_Result((float("nan"), 200.0, 260.0, 300.0))]
        return super().track(frame, **kwargs)


def test_nan_bbox_sample_is_skipped_and_job_completes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    front = tmp_path / "2026-05-25_18-30-02-front.mp4"
    write_moving_car_clip(front)
    monkeypatch.setattr("backend.app.analysis.pipeline.yolo_available", lambda: True)
    monkeypatch.setattr(
        "backend.app.analysis.pipeline.build_model",
        lambda _model_name: _NaNOnceDetector(),
    )

    result = analyze_segment("seg", front, model_name="yolo26s.pt", sample_rate_fps=5)

    assert len(result.events) == 1
    event_samples = result.events[0].samples
    assert len(event_samples) == 19  # 20 sampled frames minus the NaN-box frame
    assert all(
        math.isfinite(value) for sample in event_samples for value in sample.bbox_xyxy
    )
