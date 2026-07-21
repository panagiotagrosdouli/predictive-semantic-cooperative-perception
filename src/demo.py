"""Run a deterministic end-to-end prototype without downloading OPV2V."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from .core import (
    TrackedObject,
    critical_recall,
    estimate_future_risk,
    predict_constant_velocity,
    schedule,
    score_priority,
)


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def synthetic_objects(count: int, seed: int) -> list[TrackedObject]:
    rng = random.Random(seed)
    objects: list[TrackedObject] = []
    for index in range(count):
        x = rng.uniform(4.0, 45.0)
        y = rng.uniform(-9.0, 9.0)
        # Most objects move slowly; a subset closes rapidly on the ego vehicle.
        vx = rng.uniform(-8.0, 2.0)
        vy = rng.uniform(-1.5, 1.5)
        objects.append(
            TrackedObject(
                object_id=f"vehicle-{index:03d}",
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                uncertainty=rng.uniform(0.02, 0.45),
            )
        )
    return objects


def run(config: dict) -> None:
    prediction_cfg = config["prediction"]
    risk_cfg = config["risk"]
    scheduler_cfg = config["scheduler"]
    simulation_cfg = config["simulation"]

    processed: list[TrackedObject] = []
    for obj in synthetic_objects(
        count=int(simulation_cfg["num_objects"]),
        seed=int(simulation_cfg["random_seed"]),
    ):
        predicted = predict_constant_velocity(obj, float(prediction_cfg["horizon_seconds"]))
        risky = estimate_future_risk(
            predicted,
            critical_distance_m=float(risk_cfg["critical_distance_m"]),
            ttc_cap_s=float(risk_cfg["time_to_collision_cap_s"]),
        )
        processed.append(score_priority(risky, scheduler_cfg["weights"]))

    selected = schedule(
        processed,
        budget=int(scheduler_cfg["message_budget"]),
        strategy=str(scheduler_cfg["strategy"]),
    )

    print("Predictive semantic scheduling demo")
    print(f"Objects observed: {len(processed)}")
    print(f"Objects transmitted: {len(selected)}")
    print(f"Communication reduction: {1.0 - len(selected) / len(processed):.1%}")
    print(f"Critical-object recall: {critical_recall(processed, selected):.1%}")
    print("\nSelected objects:")
    for obj in selected:
        print(
            f"  {obj.object_id}: risk={obj.risk:.3f}, "
            f"priority={obj.priority:.3f}, distance={obj.distance:.1f} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/opv2v.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
