from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackSample:
    track_id: int
    frame_index: int
    timestamp_s: float
    bbox_xyxy: BBox
    class_name: str
    frame_width: int
    frame_height: int

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, y2)


@dataclass
class SignalSeries:
    timestamps_s: list[float] = field(default_factory=list)
    left_energy: list[float] = field(default_factory=list)
    right_energy: list[float] = field(default_factory=list)


@dataclass
class EventCandidate:
    track_id: int
    vehicle_type: str
    start_s: float
    end_s: float
    key_s: float
    lane_change_score: float
    turn_signal_score: float
    confidence: float
    reason_labels: list[str]
    samples: list[TrackSample] = field(default_factory=list)
