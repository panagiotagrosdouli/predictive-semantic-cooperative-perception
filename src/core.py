"""Core models for predictive, risk-aware semantic scheduling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Iterable


@dataclass(frozen=True)
class TrackedObject:
    """Minimal object state in the ego-vehicle coordinate frame."""

    object_id: str
    x: float
    y: float
    vx: float
    vy: float
    uncertainty: float = 0.0
    predicted_x: float | None = None
    predicted_y: float | None = None
    risk: float = 0.0
    priority: float = 0.0

    @property
    def distance(self) -> float:
        return hypot(self.x, self.y)

    @property
    def closing_speed(self) -> float:
        distance = max(self.distance, 1e-9)
        radial_velocity = (self.x * self.vx + self.y * self.vy) / distance
        return max(0.0, -radial_velocity)


def predict_constant_velocity(obj: TrackedObject, horizon_seconds: float) -> TrackedObject:
    """Predict a future position using a constant-velocity model."""

    if horizon_seconds < 0:
        raise ValueError("horizon_seconds must be non-negative")
    return replace(
        obj,
        predicted_x=obj.x + obj.vx * horizon_seconds,
        predicted_y=obj.y + obj.vy * horizon_seconds,
    )


def estimate_future_risk(
    obj: TrackedObject,
    critical_distance_m: float,
    ttc_cap_s: float,
) -> TrackedObject:
    """Estimate a bounded future-risk score in [0, 1]."""

    if critical_distance_m <= 0 or ttc_cap_s <= 0:
        raise ValueError("risk parameters must be positive")

    px = obj.predicted_x if obj.predicted_x is not None else obj.x
    py = obj.predicted_y if obj.predicted_y is not None else obj.y
    future_distance = hypot(px, py)
    proximity = max(0.0, 1.0 - future_distance / critical_distance_m)

    if obj.closing_speed > 0:
        ttc = obj.distance / obj.closing_speed
        urgency = max(0.0, 1.0 - min(ttc, ttc_cap_s) / ttc_cap_s)
    else:
        urgency = 0.0

    risk = min(1.0, 0.65 * proximity + 0.35 * urgency)
    return replace(obj, risk=risk)


def score_priority(obj: TrackedObject, weights: dict[str, float]) -> TrackedObject:
    """Calculate semantic transmission priority from normalized signals."""

    distance_signal = 1.0 / (1.0 + obj.distance)
    closing_signal = obj.closing_speed / (1.0 + obj.closing_speed)
    uncertainty_signal = max(0.0, min(1.0, obj.uncertainty))
    priority = (
        weights.get("collision_risk", 0.0) * obj.risk
        + weights.get("proximity", 0.0) * distance_signal
        + weights.get("uncertainty", 0.0) * uncertainty_signal
        + weights.get("closing_speed", 0.0) * closing_signal
    )
    return replace(obj, priority=priority)


def schedule(objects: Iterable[TrackedObject], budget: int, strategy: str) -> list[TrackedObject]:
    """Select objects under a message-count communication budget."""

    candidates = list(objects)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if strategy == "all":
        return candidates
    if strategy == "nearest":
        ranked = sorted(candidates, key=lambda item: item.distance)
    elif strategy == "predictive_risk":
        ranked = sorted(candidates, key=lambda item: item.priority, reverse=True)
    elif strategy == "random":
        # Deterministic order keeps experiments reproducible; the demo shuffles
        # candidates with a seeded generator before calling this strategy.
        ranked = candidates
    else:
        raise ValueError(f"unknown scheduling strategy: {strategy}")
    return ranked[:budget]


def critical_recall(all_objects: Iterable[TrackedObject], selected: Iterable[TrackedObject], threshold: float = 0.5) -> float:
    """Return recall of objects whose estimated future risk exceeds threshold."""

    critical = {obj.object_id for obj in all_objects if obj.risk >= threshold}
    if not critical:
        return 1.0
    sent = {obj.object_id for obj in selected}
    return len(critical & sent) / len(critical)
