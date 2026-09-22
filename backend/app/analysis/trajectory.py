from __future__ import annotations

from collections import defaultdict
from statistics import median

from .signals import detect_turn_signal
from .types import EventCandidate, SignalSeries, TrackSample

LATERAL_SHIFT_THRESHOLD = 0.12
LANE_CROSS_BONUS = 0.15
MIN_DURATION_S = 1.0


def detect_lane_change_events(
    samples: list[TrackSample],
    signal_series_by_track: dict[int, SignalSeries] | None = None,
    *,
    min_duration_s: float = MIN_DURATION_S,
    lateral_shift_threshold: float = LATERAL_SHIFT_THRESHOLD,
    lane_center_by_timestamp: dict[float, float] | None = None,
) -> list[EventCandidate]:
    by_track: dict[int, list[TrackSample]] = defaultdict(list)
    for sample in samples:
        by_track[sample.track_id].append(sample)

    events: list[EventCandidate] = []
    signal_series_by_track = signal_series_by_track or {}
    for track_id, track_samples in by_track.items():
        track_samples = sorted(track_samples, key=lambda item: item.timestamp_s)
        if len(track_samples) < 6:
            continue

        duration = track_samples[-1].timestamp_s - track_samples[0].timestamp_s
        if duration < min_duration_s:
            continue

        positions, lane_hits = _lateral_positions(track_samples, lane_center_by_timestamp)
        group = max(2, len(track_samples) // 5)
        first_pos = median(positions[:group])
        last_pos = median(positions[-group:])
        shift_norm = abs(last_pos - first_pos)
        used_geometry = lane_hits >= len(track_samples) * 0.5

        lower_half_ratio = sum(
            1 for sample in track_samples if sample.bottom_center[1] > sample.frame_height * 0.45
        ) / len(track_samples)
        if shift_norm < lateral_shift_threshold or lower_half_ratio < 0.55:
            continue

        lane_change_score = min(1.0, (shift_norm - lateral_shift_threshold) / 0.22 + 0.35)
        crossed = used_geometry and first_pos * last_pos < 0
        if crossed:
            lane_change_score = min(1.0, lane_change_score + LANE_CROSS_BONUS)
        start_s = max(0.0, track_samples[0].timestamp_s - 1.5)
        end_s = track_samples[-1].timestamp_s + 1.5
        key_s = track_samples[len(track_samples) // 2].timestamp_s

        turn_signal_score, signal_label = detect_turn_signal(
            signal_series_by_track.get(track_id, SignalSeries()),
            max(0.0, key_s - 2.5),
            key_s + 2.5,
        )
        suspicious_signal = 1.0 - turn_signal_score
        confidence = round(max(0.0, min(1.0, lane_change_score * suspicious_signal)), 3)
        if signal_label == "turn_signal_inconclusive":
            confidence = round(confidence * 0.72, 3)

        labels = ["lane_change_candidate", signal_label]
        if last_pos > first_pos:
            labels.append("moving_right")
        else:
            labels.append("moving_left")
        if crossed:
            labels.append("lane_geometry")
        elif used_geometry:
            labels.append("ego_motion_compensated")

        if signal_label == "turn_signal_observed":
            continue

        events.append(
            EventCandidate(
                track_id=track_id,
                vehicle_type=track_samples[-1].class_name,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                key_s=round(key_s, 3),
                lane_change_score=round(lane_change_score, 3),
                turn_signal_score=round(turn_signal_score, 3),
                confidence=confidence,
                reason_labels=labels,
                samples=track_samples,
            )
        )

    return merge_overlapping_events(events)


def _lateral_positions(
    track_samples: list[TrackSample],
    lane_center_by_timestamp: dict[float, float] | None,
) -> tuple[list[float], int]:
    """Normalized lateral position. Lane centers subtract ego-motion from the shift."""

    positions: list[float] = []
    lane_hits = 0
    for sample in track_samples:
        frame_width = max(1, sample.frame_width)
        center = None if lane_center_by_timestamp is None else lane_center_by_timestamp.get(
            sample.timestamp_s
        )
        if center is None:
            positions.append(sample.bottom_center[0] / frame_width)
            continue
        lane_hits += 1
        positions.append((sample.bottom_center[0] - center) / frame_width)
    return positions, lane_hits


def merge_overlapping_events(events: list[EventCandidate]) -> list[EventCandidate]:
    ordered = sorted(events, key=lambda event: (event.track_id, event.start_s, -event.confidence))
    merged: list[EventCandidate] = []
    for event in ordered:
        if not merged:
            merged.append(event)
            continue
        prev = merged[-1]
        overlaps = event.track_id == prev.track_id and event.start_s <= prev.end_s + 1.0
        if not overlaps:
            merged.append(event)
            continue
        if event.confidence > prev.confidence:
            event.start_s = min(prev.start_s, event.start_s)
            event.end_s = max(prev.end_s, event.end_s)
            merged[-1] = event
        else:
            prev.end_s = max(prev.end_s, event.end_s)
    return merged
