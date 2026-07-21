"""Lightweight OPV2V metadata loader.

The loader intentionally reads only per-frame YAML metadata. It does not require
Open3D, PyTorch, or the full OpenCOOD stack, which keeps the scheduling prototype
small and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from src.core import TrackedObject


@dataclass(frozen=True)
class OPV2VFrame:
    """One observer vehicle's annotations at one timestamp."""

    scenario_id: str
    observer_id: str
    timestamp: str
    ego_x: float
    ego_y: float
    objects: tuple[TrackedObject, ...]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _xy(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return _as_float(value.get("x")), _as_float(value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _as_float(value[0]), _as_float(value[1])
    return 0.0, 0.0


def _ego_xy(metadata: Mapping[str, Any]) -> tuple[float, float]:
    # OPV2V commonly stores [x, y, z, roll, yaw, pitch] in lidar_pose.
    for key in ("lidar_pose", "true_ego_pos", "ego_pos"):
        if key in metadata:
            return _xy(metadata[key])
    return 0.0, 0.0


def _yaw_degrees(annotation: Mapping[str, Any]) -> float:
    angle = annotation.get("angle", annotation.get("rotation", 0.0))
    if isinstance(angle, Mapping):
        return _as_float(angle.get("yaw", angle.get("z", 0.0)))
    if isinstance(angle, (list, tuple)):
        # CARLA/OpenCOOD annotations usually keep yaw as the second element.
        if len(angle) >= 2:
            return _as_float(angle[1])
        if angle:
            return _as_float(angle[0])
    return _as_float(angle)


def _velocity_xy(annotation: Mapping[str, Any]) -> tuple[float, float]:
    velocity = annotation.get("velocity")
    if isinstance(velocity, Mapping) or isinstance(velocity, (list, tuple)):
        return _xy(velocity)

    # Some dumps expose scalar speed. Treat values as km/h, matching the OPV2V
    # ego-speed convention, and resolve them along the annotated yaw.
    speed_kmh = _as_float(annotation.get("speed", velocity), 0.0)
    speed_ms = speed_kmh / 3.6
    yaw = radians(_yaw_degrees(annotation))
    return speed_ms * cos(yaw), speed_ms * sin(yaw)


def parse_metadata(
    metadata: Mapping[str, Any],
    *,
    scenario_id: str,
    observer_id: str,
    timestamp: str,
) -> OPV2VFrame:
    """Convert one OPV2V YAML dictionary into scheduler-ready objects.

    Object positions are translated to the observer origin. This first version
    assumes world axes and ego axes are aligned; explicit yaw rotation can be
    added later when evaluating exact geometric fusion.
    """

    ego_x, ego_y = _ego_xy(metadata)
    annotations = metadata.get("vehicles", metadata.get("objects", {}))
    objects: list[TrackedObject] = []

    if isinstance(annotations, Mapping):
        items = annotations.items()
    elif isinstance(annotations, list):
        items = ((str(i), item) for i, item in enumerate(annotations))
    else:
        items = ()

    for object_id, raw in items:
        if not isinstance(raw, Mapping):
            continue
        world_x, world_y = _xy(raw.get("location", raw.get("center")))
        vx, vy = _velocity_xy(raw)
        objects.append(
            TrackedObject(
                object_id=str(object_id),
                x=world_x - ego_x,
                y=world_y - ego_y,
                vx=vx,
                vy=vy,
                uncertainty=_as_float(raw.get("uncertainty"), 0.0),
            )
        )

    return OPV2VFrame(
        scenario_id=scenario_id,
        observer_id=observer_id,
        timestamp=timestamp,
        ego_x=ego_x,
        ego_y=ego_y,
        objects=tuple(objects),
    )


def load_metadata_file(path: str | Path) -> OPV2VFrame:
    """Load one ``<timestamp>.yaml`` file from an OPV2V agent directory."""

    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    if yaml_path.name == "data_protocol.yaml":
        raise ValueError("data_protocol.yaml is scenario metadata, not a frame")

    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"expected a YAML mapping in {yaml_path}")

    return parse_metadata(
        metadata,
        scenario_id=yaml_path.parent.parent.name,
        observer_id=yaml_path.parent.name,
        timestamp=yaml_path.stem,
    )


def iter_agent_frames(agent_dir: str | Path) -> Iterator[OPV2VFrame]:
    """Yield timestamp-ordered YAML frames for one connected vehicle."""

    directory = Path(agent_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    for path in sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9].yaml")):
        yield load_metadata_file(path)


def discover_agents(split_dir: str | Path) -> list[Path]:
    """Find all numeric agent directories under an OPV2V split."""

    root = Path(split_dir)
    agents: list[Path] = []
    for scenario in sorted(path for path in root.iterdir() if path.is_dir()):
        agents.extend(
            sorted(
                path
                for path in scenario.iterdir()
                if path.is_dir() and path.name.lstrip("-").isdigit()
            )
        )
    return agents
