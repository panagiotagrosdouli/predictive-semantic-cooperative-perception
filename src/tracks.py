"""Temporal track construction for OPV2V annotation frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.core import TrackedObject
from src.opv2v import OPV2VFrame


@dataclass(frozen=True)
class TrackPoint:
    timestamp: str
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class ObjectTrack:
    object_id: str
    points: tuple[TrackPoint, ...]

    @property
    def length(self) -> int:
        return len(self.points)


def build_tracks(frames: Iterable[OPV2VFrame], min_length: int = 2) -> list[ObjectTrack]:
    """Group objects with the same ID across timestamp-ordered frames."""

    if min_length < 1:
        raise ValueError("min_length must be at least 1")

    grouped: dict[str, list[TrackPoint]] = {}
    for frame in sorted(frames, key=lambda item: item.timestamp):
        for obj in frame.objects:
            grouped.setdefault(obj.object_id, []).append(
                TrackPoint(frame.timestamp, obj.x, obj.y, obj.vx, obj.vy)
            )

    return [
        ObjectTrack(object_id, tuple(points))
        for object_id, points in sorted(grouped.items())
        if len(points) >= min_length
    ]


def latest_state(track: ObjectTrack, dt_seconds: float | None = None) -> TrackedObject:
    """Return the latest scheduler state, optionally estimating velocity from positions."""

    if not track.points:
        raise ValueError("track must contain at least one point")

    last = track.points[-1]
    vx, vy = last.vx, last.vy
    if dt_seconds is not None:
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        if len(track.points) >= 2:
            previous = track.points[-2]
            vx = (last.x - previous.x) / dt_seconds
            vy = (last.y - previous.y) / dt_seconds

    return TrackedObject(track.object_id, last.x, last.y, vx, vy)
