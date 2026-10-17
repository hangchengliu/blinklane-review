from __future__ import annotations

from collections import defaultdict
from statistics import median

from .signals import detect_turn_signal
from .types import EventCandidate, SignalSeries, TrackSample

LATERAL_SHIFT_THRESHOLD = 0.12
LANE_CROSS_BONUS = 0.15
MIN_DURATION_S = 1.0
MIN_TRACK_SAMPLES = 6
# Front-camera corroboration floor for an "observed" label; matches the
# flat-dark bound in signals.detect_turn_signal (both assume the *8.0 gain).
FRONT_CORROBORATION_MIN = 0.12


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
        if len(track_samples) < MIN_TRACK_SAMPLES:
            continue

        duration = track_samples[-1].timestamp_s - track_samples[0].timestamp_s
        if duration < min_duration_s:
            continue

        pos_samples, positions, used_geometry = _lateral_positions(
            track_samples, lane_center_by_timestamp, min_duration_s
        )
        group = max(2, len(positions) // 5)
        first_pos = median(positions[:group])
        last_pos = median(positions[-group:])
        shift_norm = abs(last_pos - first_pos)

        # 0.45/0.55 gates: the target must spend most of the track in the lower
        # 45% of the frame (near-field vehicles); tuned on synthetic fixtures to
        # reject distant traffic drifting across the horizon.
        lower_half_ratio = sum(
            1 for sample in pos_samples if sample.bottom_center[1] > sample.frame_height * 0.45
        ) / len(pos_samples)
        if shift_norm < lateral_shift_threshold or lower_half_ratio < 0.55:
            continue

        # 0.22/0.35 shaping: shift of exactly the threshold scores 0.35 and each
        # additional 0.22 of normalized shift saturates the score; empirically
        # tuned on synthetic fixtures, interacts with LATERAL_SHIFT_THRESHOLD.
        lane_change_score = min(1.0, (shift_norm - lateral_shift_threshold) / 0.22 + 0.35)
        crossed = used_geometry and first_pos * last_pos < 0
        if crossed:
            lane_change_score = min(1.0, lane_change_score + LANE_CROSS_BONUS)
        start_s = max(0.0, track_samples[0].timestamp_s - 1.5)
        end_s = track_samples[-1].timestamp_s + 1.5
        key_s = track_samples[len(track_samples) // 2].timestamp_s

        # The scoring window must cover the whole event, not just key_s +/- 2.5:
        # blinks near the start or end of a long maneuver still count.
        window_start = min(max(0.0, key_s - 2.5), start_s)
        window_end = max(key_s + 2.5, end_s)
        series = signal_series_by_track.get(track_id, SignalSeries())
        turn_signal_score, signal_label = detect_turn_signal(series, window_start, window_end)
        if signal_label == "turn_signal_observed" and not _front_camera_corroborates(
            series, window_start, window_end
        ):
            # Repeater energy is real evidence and keeps its score, but a blink
            # seen ONLY on a repeater (e.g. an unrelated side vehicle) must not
            # by itself clear the event; cap the label at inconclusive.
            signal_label = "turn_signal_inconclusive"
        suspicious_signal = 1.0 - turn_signal_score
        confidence = round(max(0.0, min(1.0, lane_change_score * suspicious_signal)), 3)
        if signal_label == "turn_signal_inconclusive":
            # 0.72 inconclusive penalty: uncertain signal evidence keeps the
            # event but discounts confidence; tuned so clear no-signal events
            # still outrank inconclusive ones.
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

    return events


def _lateral_positions(
    track_samples: list[TrackSample],
    lane_center_by_timestamp: dict[float, float] | None,
    min_duration_s: float,
) -> tuple[list[TrackSample], list[float], bool]:
    """Normalized lateral positions in ONE reference frame per track.

    Lane-relative positions (ego-motion subtracted) are centered near 0 while
    absolute positions sit near 0.3-0.7, so mixing both in one series invents
    phantom shifts when lane detection flickers. If enough samples have a lane
    center, only those samples are used (single lane-relative frame); otherwise
    the whole track falls back to absolute coordinates.
    """

    hits: list[tuple[TrackSample, float]] = []
    if lane_center_by_timestamp is not None:
        for sample in track_samples:
            center = lane_center_by_timestamp.get(sample.timestamp_s)
            if center is not None:
                hits.append((sample, center))
    enough_hits = (
        len(hits) >= MIN_TRACK_SAMPLES
        and hits[-1][0].timestamp_s - hits[0][0].timestamp_s >= min_duration_s
    )
    if enough_hits:
        samples = [sample for sample, _center in hits]
        positions = [
            (sample.bottom_center[0] - center) / max(1, sample.frame_width)
            for sample, center in hits
        ]
        return samples, positions, True
    positions = [sample.bottom_center[0] / max(1, sample.frame_width) for sample in track_samples]
    return track_samples, positions, False


def _front_camera_corroborates(series: SignalSeries, start_s: float, end_s: float) -> bool:
    """True when the front camera itself saw amber energy in the window.

    Uses the pre-merge front snapshot when a repeater was merged in; otherwise
    left/right ARE the front camera and are used directly.
    """

    front = series.front_energy or [
        max(left, right)
        for left, right in zip(series.left_energy, series.right_energy, strict=False)
    ]
    peak = 0.0
    for timestamp, energy in zip(series.timestamps_s, front, strict=False):
        if start_s <= timestamp <= end_s:
            peak = max(peak, energy)
    return peak >= FRONT_CORROBORATION_MIN
