from src.core import (
    TrackedObject,
    critical_recall,
    estimate_future_risk,
    predict_constant_velocity,
    schedule,
    score_priority,
)


def test_constant_velocity_prediction() -> None:
    obj = TrackedObject("a", x=10.0, y=2.0, vx=-2.0, vy=1.0)
    predicted = predict_constant_velocity(obj, 2.0)
    assert predicted.predicted_x == 6.0
    assert predicted.predicted_y == 4.0


def test_predictive_scheduler_prefers_high_priority() -> None:
    weights = {"collision_risk": 1.0}
    low = score_priority(TrackedObject("low", 20, 0, 0, 0, risk=0.1), weights)
    high = score_priority(TrackedObject("high", 5, 0, -2, 0, risk=0.9), weights)
    selected = schedule([low, high], budget=1, strategy="predictive_risk")
    assert selected[0].object_id == "high"


def test_future_risk_is_bounded() -> None:
    obj = predict_constant_velocity(TrackedObject("a", 5, 0, -3, 0), 1.0)
    result = estimate_future_risk(obj, critical_distance_m=6.0, ttc_cap_s=8.0)
    assert 0.0 <= result.risk <= 1.0


def test_critical_recall() -> None:
    objects = [
        TrackedObject("critical", 2, 0, 0, 0, risk=0.8),
        TrackedObject("safe", 20, 0, 0, 0, risk=0.1),
    ]
    assert critical_recall(objects, [objects[0]]) == 1.0
    assert critical_recall(objects, [objects[1]]) == 0.0
