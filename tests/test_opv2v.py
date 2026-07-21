from pathlib import Path

import yaml

from src.opv2v import discover_agents, iter_agent_frames, load_metadata_file, parse_metadata


def test_parse_metadata_translates_objects_to_ego_frame() -> None:
    frame = parse_metadata(
        {
            "lidar_pose": [100.0, 50.0, 2.0, 0.0, 0.0, 0.0],
            "vehicles": {
                17: {
                    "location": [106.0, 47.0, 0.0],
                    "velocity": [-2.0, 1.5, 0.0],
                }
            },
        },
        scenario_id="scenario",
        observer_id="1732",
        timestamp="00000",
    )
    obj = frame.objects[0]
    assert obj.object_id == "17"
    assert obj.x == 6.0
    assert obj.y == -3.0
    assert obj.vx == -2.0
    assert obj.vy == 1.5


def test_scalar_speed_is_converted_from_kmh() -> None:
    frame = parse_metadata(
        {
            "ego_pos": [0.0, 0.0],
            "vehicles": {
                "car": {
                    "center": [10.0, 0.0],
                    "speed": 36.0,
                    "angle": [0.0, 0.0, 0.0],
                }
            },
        },
        scenario_id="s",
        observer_id="a",
        timestamp="00001",
    )
    assert round(frame.objects[0].vx, 6) == 10.0
    assert round(frame.objects[0].vy, 6) == 0.0


def test_file_discovery_and_iteration(tmp_path: Path) -> None:
    agent = tmp_path / "train" / "scenario_a" / "1732"
    agent.mkdir(parents=True)
    for timestamp in ("00001", "00000"):
        with (agent / f"{timestamp}.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump({"lidar_pose": [0, 0], "vehicles": {}}, stream)

    agents = discover_agents(tmp_path / "train")
    assert agents == [agent]
    assert [frame.timestamp for frame in iter_agent_frames(agent)] == ["00000", "00001"]
    assert load_metadata_file(agent / "00000.yaml").observer_id == "1732"
