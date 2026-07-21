"""Run communication-scheduling baselines on OPV2V annotation tracks."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import yaml

from src.core import (
    TrackedObject,
    critical_recall,
    estimate_future_risk,
    predict_constant_velocity,
    schedule,
    score_priority,
)
from src.opv2v import discover_agents, iter_agent_frames
from src.tracks import build_tracks, latest_state


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    objects_observed: int
    objects_transmitted: int
    communication_reduction: float
    critical_recall: float
    retained_priority: float


def prepare_objects(
    objects: Iterable[TrackedObject],
    *,
    horizon_seconds: float,
    critical_distance_m: float,
    ttc_cap_s: float,
    weights: dict[str, float],
) -> list[TrackedObject]:
    """Apply prediction, risk estimation and semantic scoring."""

    prepared: list[TrackedObject] = []
    for obj in objects:
        item = predict_constant_velocity(obj, horizon_seconds)
        item = estimate_future_risk(item, critical_distance_m, ttc_cap_s)
        item = score_priority(item, weights)
        prepared.append(item)
    return prepared


def evaluate_strategy(
    objects: list[TrackedObject],
    *,
    strategy: str,
    budget: int,
    seed: int = 7,
) -> StrategyResult:
    """Evaluate one scheduler on one set of scored objects."""

    candidates = list(objects)
    if strategy == "random":
        random.Random(seed).shuffle(candidates)

    selected = schedule(candidates, budget=budget, strategy=strategy)
    observed = len(objects)
    transmitted = len(selected)
    reduction = 0.0 if observed == 0 else 1.0 - transmitted / observed
    total_priority = sum(item.priority for item in objects)
    selected_priority = sum(item.priority for item in selected)
    retained = 1.0 if total_priority == 0 else selected_priority / total_priority

    return StrategyResult(
        strategy=strategy,
        objects_observed=observed,
        objects_transmitted=transmitted,
        communication_reduction=reduction,
        critical_recall=critical_recall(objects, selected),
        retained_priority=retained,
    )


def run_agent_experiment(
    agent_dir: str | Path,
    *,
    max_frames: int,
    timestep_seconds: float,
    horizon_seconds: float,
    critical_distance_m: float,
    ttc_cap_s: float,
    weights: dict[str, float],
    budget: int,
    seed: int,
) -> list[StrategyResult]:
    """Build tracks for one observer and compare all scheduling strategies."""

    frames = list(iter_agent_frames(agent_dir))[:max_frames]
    tracks = build_tracks(frames, min_length=2)
    objects = [latest_state(track, dt_seconds=timestep_seconds) for track in tracks]
    prepared = prepare_objects(
        objects,
        horizon_seconds=horizon_seconds,
        critical_distance_m=critical_distance_m,
        ttc_cap_s=ttc_cap_s,
        weights=weights,
    )
    return [
        evaluate_strategy(prepared, strategy=strategy, budget=budget, seed=seed)
        for strategy in ("all", "nearest", "random", "predictive_risk")
    ]


def aggregate(results: Iterable[StrategyResult]) -> list[StrategyResult]:
    """Average per-agent metrics by strategy."""

    grouped: dict[str, list[StrategyResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy, []).append(result)

    output: list[StrategyResult] = []
    for strategy in ("all", "nearest", "random", "predictive_risk"):
        rows = grouped.get(strategy, [])
        if not rows:
            continue
        output.append(
            StrategyResult(
                strategy=strategy,
                objects_observed=round(mean(row.objects_observed for row in rows)),
                objects_transmitted=round(mean(row.objects_transmitted for row in rows)),
                communication_reduction=mean(row.communication_reduction for row in rows),
                critical_recall=mean(row.critical_recall for row in rows),
                retained_priority=mean(row.retained_priority for row in rows),
            )
        )
    return output


def save_csv(results: Iterable[StrategyResult], output_path: str | Path) -> None:
    """Write aggregate experiment metrics to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(StrategyResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/opv2v.yaml")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--max-agents", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--output", default="results/opv2v_baselines.csv")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    dataset = config["dataset"]
    split_dir = Path(args.split_dir or Path(dataset["root"]) / dataset["split"])
    agents = discover_agents(split_dir)[: args.max_agents]
    if not agents:
        raise SystemExit(f"No OPV2V agent directories found under {split_dir}")

    prediction = config["prediction"]
    risk = config["risk"]
    scheduler_config = config["scheduler"]
    seed = config.get("simulation", {}).get("random_seed", 7)

    all_results: list[StrategyResult] = []
    for agent in agents:
        all_results.extend(
            run_agent_experiment(
                agent,
                max_frames=args.max_frames,
                timestep_seconds=float(prediction["timestep_seconds"]),
                horizon_seconds=float(prediction["horizon_seconds"]),
                critical_distance_m=float(risk["critical_distance_m"]),
                ttc_cap_s=float(risk["time_to_collision_cap_s"]),
                weights=dict(scheduler_config["weights"]),
                budget=int(scheduler_config["message_budget"]),
                seed=int(seed),
            )
        )

    summary = aggregate(all_results)
    save_csv(summary, args.output)

    print(f"Agents evaluated: {len(agents)}")
    print(f"Results written to: {args.output}")
    print("strategy          sent/seen   reduction   critical recall   retained priority")
    for row in summary:
        print(
            f"{row.strategy:16} {row.objects_transmitted:3}/{row.objects_observed:<3} "
            f"{row.communication_reduction:9.1%} {row.critical_recall:17.1%} "
            f"{row.retained_priority:19.1%}"
        )


if __name__ == "__main__":
    main()
