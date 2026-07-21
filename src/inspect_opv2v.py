"""Inspect OPV2V YAML annotations without loading point clouds."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.opv2v import discover_agents, iter_agent_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path, help="Path such as datasets/opv2v/train")
    parser.add_argument("--max-agents", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agents = discover_agents(args.split_dir)
    if not agents:
        raise SystemExit(f"No numeric OPV2V agent directories found under {args.split_dir}")

    object_counts: Counter[str] = Counter()
    total_frames = 0
    total_objects = 0

    for agent_dir in agents[: args.max_agents]:
        print(f"\nAgent: {agent_dir}")
        for index, frame in enumerate(iter_agent_frames(agent_dir)):
            if index >= args.max_frames:
                break
            total_frames += 1
            total_objects += len(frame.objects)
            object_counts.update(obj.object_id for obj in frame.objects)
            print(
                f"  t={frame.timestamp} ego=({frame.ego_x:.2f}, {frame.ego_y:.2f}) "
                f"objects={len(frame.objects)}"
            )
            for obj in frame.objects[:3]:
                print(
                    f"    {obj.object_id}: rel=({obj.x:.2f}, {obj.y:.2f}) "
                    f"v=({obj.vx:.2f}, {obj.vy:.2f})"
                )

    repeated = sum(count > 1 for count in object_counts.values())
    print("\nSummary")
    print(f"  discovered agents: {len(agents)}")
    print(f"  inspected frames: {total_frames}")
    print(f"  annotation instances: {total_objects}")
    print(f"  object IDs repeated across inspected frames: {repeated}")


if __name__ == "__main__":
    main()
