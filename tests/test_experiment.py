from src.core import TrackedObject
from src.experiment import aggregate, evaluate_strategy, prepare_objects


def test_prepare_objects_adds_prediction_risk_and_priority() -> None:
    objects = [TrackedObject("a", x=4.0, y=0.0, vx=-1.0, vy=0.0)]
    prepared = prepare_objects(
        objects,
        horizon_seconds=2.0,
        critical_distance_m=6.0,
        ttc_cap_s=8.0,
        weights={"collision_risk": 1.0},
    )
    assert prepared[0].predicted_x == 2.0
    assert prepared[0].risk > 0.0
    assert prepared[0].priority == prepared[0].risk


def test_predictive_strategy_retains_high_priority_object() -> None:
    objects = [
        TrackedObject("low", 20, 0, 0, 0, risk=0.1, priority=0.1),
        TrackedObject("high", 5, 0, -1, 0, risk=0.9, priority=0.9),
    ]
    result = evaluate_strategy(objects, strategy="predictive_risk", budget=1)
    assert result.objects_transmitted == 1
    assert result.communication_reduction == 0.5
    assert result.critical_recall == 1.0
    assert result.retained_priority == 0.9


def test_aggregate_groups_strategies() -> None:
    objects = [TrackedObject("a", 2, 0, 0, 0, risk=0.8, priority=1.0)]
    rows = [
        evaluate_strategy(objects, strategy="all", budget=1),
        evaluate_strategy(objects, strategy="all", budget=1),
    ]
    summary = aggregate(rows)
    assert len(summary) == 1
    assert summary[0].strategy == "all"
    assert summary[0].critical_recall == 1.0
